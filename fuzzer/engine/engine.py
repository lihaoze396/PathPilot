#!/usr/bin/env python3
# -*- coding: utf-8 -*-

''' Genetic Algorithm engine definition '''

import math
import random
import sys
import time
import logging

from functools import wraps

# Imports for profiling.
import cProfile
import pstats
import os

# 鑾峰彇鏍圭洰褰?BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/../'
# 灏嗘牴鐩綍娣诲姞鍒皃ath涓?sys.path.append(BASE_DIR)
from fuzzer.utils import settings
try:
    from fuzzer.runtime.runtime_timing import budget_exhausted, has_time_budget, record_wall_snapshot_if_needed
except ImportError:
    from fuzzer.runtime.runtime_timing import budget_exhausted, has_time_budget, record_wall_snapshot_if_needed

from .components import Individual, Population
from .plugin_interfaces.operators import Selection, Crossover, Mutation
from .plugin_interfaces.analysis import OnTheFlyAnalysis


def do_profile(filename, sortby='tottime'):
    '''
    Constructor for function profiling decorator.
    '''

    def _do_profile(func):
        '''
        Function profiling decorator.
        '''

        @wraps(func)
        def profiled_func(*args, **kwargs):
            '''
            Decorated function.
            '''
            # Flag for doing profiling or not.
            DO_PROF = os.getenv('PROFILING')

            if DO_PROF:
                profile = cProfile.Profile()
                profile.enable()
                result = func(*args, **kwargs)
                profile.disable()
                ps = pstats.Stats(profile).sort_stats(sortby)
                ps.dump_stats(filename)
            else:
                result = func(*args, **kwargs)
            return result

        return profiled_func

    return _do_profile


class StatVar(object):
    def __init__(self, name):
        '''
        Descriptor for statistical variables which need to be memoized when
        engine is running.
        '''
        # Protected.
        self.name = '_{}'.format(name)

    def __get__(self, engine, cls):
        '''
        Getter.
        '''
        stat_var = getattr(engine, self.name)
        if stat_var is None:
            if 'min' in self.name and 'ori' in self.name:
                stat_var = engine.population.min(engine.ori_fitness)
            elif 'min' in self.name:
                stat_var = engine.population.min(engine.fitness)
            elif 'max' in self.name and 'ori' in self.name:
                stat_var = engine.population.max(engine.ori_fitness)
            elif 'max' in self.name:
                stat_var = engine.population.max(engine.fitness)
            elif 'mean' in self.name and 'ori' in self.name:
                stat_var = engine.population.mean(engine.ori_fitness)
            elif 'mean' in self.name:
                stat_var = engine.population.mean(engine.fitness)
            setattr(engine, self.name, stat_var)
        return stat_var

    def __set__(self, engine, value):
        '''
        Setter.
        '''
        setattr(engine, self.name, value)


class EvolutionaryFuzzingEngine(object):
    # Statistical attributes for population.
    fmax, fmin, fmean = StatVar('fmax'), StatVar('fmin'), StatVar('fmean')
    ori_fmax, ori_fmin, ori_fmean = (StatVar('ori_fmax'),
                                     StatVar('ori_fmin'),
                                     StatVar('ori_fmean'))

    def __init__(self, population, selection, crossover, mutation, fitness=None, analysis=None, mapping=None):
        # Set logger.
        logger_name = 'engine.{}'.format(self.__class__.__name__)
        self.logger = logging.getLogger(logger_name)

        # Attributes assignment.
        self.population = population
        self.fitness = fitness
        self.selection = selection
        self.crossover = crossover
        self.mutation = mutation
        self.analysis = [] if analysis is None else [a() for a in analysis]
        self.mapping = mapping
        self.stage_runtime = None
        self._stage_last_code_coverage_count = 0
        self._stage_last_revert_count = 0
        self._stage_last_invalid_count = 0
        self._stage_last_tx_count = 0
        self._stage_metrics_initialized = False

        # Maxima and minima in population.
        self._fmax, self._fmin, self._fmean = None, None, None
        self._ori_fmax, self._ori_fmin, self._ori_fmean = None, None, None

        # Default fitness functions.
        self.ori_fitness = None if self.fitness is None else self.fitness

        # Store current generation number.
        self.current_generation = -1  # Starts from 0.

        # Check parameters validity.
        self._check_parameters()

    @staticmethod
    def _stage_error_counts(env):
        revert_count = 0
        invalid_count = 0
        errors = getattr(env, "results", {}).get("errors", {})
        for entries in errors.values():
            if not isinstance(entries, list):
                entries = [entries]
            for entry in entries:
                text = str(entry)
                if "REVERT" in text.upper():
                    revert_count += 1
                if "INVALID" in text.upper() or "ASSERT" in text.upper():
                    invalid_count += 1
        return revert_count, invalid_count

    def _stage_metrics(self, env):
        revert_count, invalid_count = self._stage_error_counts(env)
        return {
            "code_coverage_count": len(getattr(env, "code_coverage", set())),
            "code_coverage_total": len(getattr(env, "overall_pcs", [])),
            "revert_count": revert_count,
            "invalid_count": invalid_count,
            "tx_count": int(getattr(env, "nr_of_transactions", 0) or 0),
        }

    def _install_stage_runtime_hooks(self, analysis):
        if self.stage_runtime is None or getattr(analysis, "_stage_runtime_hooked", False):
            return
        if not hasattr(analysis, "execution_function"):
            return

        original_execution_function = analysis.execution_function
        original_finalize = analysis.finalize
        engine = self

        @wraps(original_execution_function)
        def execution_function_with_stage(individual, env):
            generation = int(getattr(env, "current_generation", getattr(engine, "current_generation", 0)) or 0)
            individual_id = "{}:{}".format(generation, getattr(individual, "hash", id(individual)))
            if not engine._stage_metrics_initialized:
                before = engine._stage_metrics(env)
                engine._stage_last_code_coverage_count = before["code_coverage_count"]
                engine._stage_last_revert_count = before["revert_count"]
                engine._stage_last_invalid_count = before["invalid_count"]
                engine._stage_last_tx_count = before["tx_count"]
                engine._stage_metrics_initialized = True
            else:
                before = {
                    "code_coverage_count": engine._stage_last_code_coverage_count,
                    "code_coverage_total": len(getattr(env, "overall_pcs", [])),
                    "revert_count": engine._stage_last_revert_count,
                    "invalid_count": engine._stage_last_invalid_count,
                    "tx_count": engine._stage_last_tx_count,
                }
            engine.stage_runtime.before_individual(generation, individual_id)
            result = original_execution_function(individual, env)
            after = engine._stage_metrics(env)
            engine._stage_last_code_coverage_count = after["code_coverage_count"]
            engine._stage_last_revert_count = after["revert_count"]
            engine._stage_last_invalid_count = after["invalid_count"]
            engine._stage_last_tx_count = after["tx_count"]
            code_delta = after["code_coverage_count"] - before["code_coverage_count"]
            feedback = {
                "code_coverage_count": after["code_coverage_count"],
                "code_coverage_total": after["code_coverage_total"],
                "code_delta": code_delta,
                "hard_gain": code_delta > 0,
                "revert_count": max(after["revert_count"] - before["revert_count"], 0),
                "invalid_count": max(after["invalid_count"] - before["invalid_count"], 0),
                "tx_count": max(after["tx_count"] - before["tx_count"], 0),
                "sequence_length": len(getattr(individual, "solution", []) or []),
            }
            engine.stage_runtime.after_individual(generation, individual_id, feedback)
            return result

        @wraps(original_finalize)
        def finalize_with_stage(population, engine):
            try:
                stage_runtime = getattr(engine, "stage_runtime", None)
                if stage_runtime is not None:
                    stage_runtime.flush_logs()
                    env = getattr(analysis, "env", None)
                    if env is not None:
                        env.results.update(stage_runtime.summary())
            except Exception:
                pass
            return original_finalize(population, engine)

        analysis.execution_function = execution_function_with_stage
        analysis.finalize = finalize_with_stage
        analysis._stage_runtime_hooked = True

    def _timeout_reached(self, execution_begin):
        env = getattr(self, "env", None)
        if env is not None:
            return bool(budget_exhausted(env, fallback_timeout=settings.GLOBAL_TIMEOUT))
        return bool(
            settings.GLOBAL_TIMEOUT
            and time.time() - execution_begin >= settings.GLOBAL_TIMEOUT
        )

    def _timeout_repair_grace_candidate_ids(self):
        env = getattr(self, "env", None)
        tracker = getattr(env, "target_focus_soft_injection_analysis", None) if env is not None else None
        runtime_getter = getattr(tracker, "_repair_runtime", None)
        runtime = runtime_getter() if callable(runtime_getter) else None
        if runtime is None:
            return [], None, tracker
        candidate_ids = runtime.activate_repair_execution_grace(
            generation=int(getattr(self, "current_generation", 0) or 0),
            reason="selected_after_budget_end",
        )
        return candidate_ids, runtime, tracker

    def _run_timeout_repair_grace(self, analysis_index, g):
        candidate_ids, runtime, tracker = self._timeout_repair_grace_candidate_ids()
        if not candidate_ids:
            return False
        next_execution = None
        for idx in range(int(analysis_index) + 1, len(self.analysis)):
            candidate = self.analysis[idx]
            if candidate.__class__.__name__ == "ExecutionTraceAnalyzer":
                next_execution = candidate
                break
        if next_execution is None:
            runtime.finalize_repair_execution_grace(default_reason="no_submit_hook_called")
            return False
        try:
            next_execution.register_step(g=g, population=self.population, engine=self)
            if tracker is not None and hasattr(tracker, "_checkpoint_observe_only"):
                try:
                    tracker._checkpoint_observe_only(int(g))
                except Exception:
                    pass
            return True
        finally:
            runtime.finalize_repair_execution_grace(default_reason="selected_after_budget_end")

    @do_profile(filename='engine_run.prof')
    def run(self, ng):
        '''
        Run the Genetic Algorithm optimization iteration with specified parameters.
        '''
        try:
            execution_begin = time.time()
            env = getattr(self, "env", None)

            if self.fitness is None:
                raise AttributeError('No fitness function in GA engine')

            # Setup analysis objects.
            for a in self.analysis:
                self._install_stage_runtime_hooks(a)
                a.setup(ng=ng, engine=self)
                a.register_step(g=-1, population=self.population, engine=self)

            # Enter evolution iteration.
            g = 0
            while g < ng or (has_time_budget(env, settings.GLOBAL_TIMEOUT) if env is not None else bool(settings.GLOBAL_TIMEOUT)):
                if env is not None:
                    record_wall_snapshot_if_needed(env)
                if self._timeout_reached(execution_begin):
                    break

                self.current_generation = g

                indvs = []
                generation_timed_out = False
                # NOTE: One series of genetic operation generates 2 new individuals.
                size = self.population.size // 3

                # Fill the new population.
                if settings.TRANS_MODE == "cross" and len(settings.TRANS_CROSS_BAD_INDVS) > 0:
                    new_pop = self.population.new()
                    new_pop.size = size
                    indv = new_pop.init()
                    if settings.TRANS_SUPPORT_MODE == 1:
                        for child in indv:
                            child.append_other()
                    indvs.extend(indv)
                size = (self.population.size - len(indvs)) // 2
                for _ in range(size):
                    if self._timeout_reached(execution_begin):
                        generation_timed_out = True
                        break
                    # Select father and mother.
                    parents = self.selection.select(self.population, fitness=self.fitness)
                    # Crossover.
                    children = self.crossover.cross(*parents)
                    # Mutation.
                    children = [self.mutation.mutate(child, self) for child in children]
                    if settings.TRANS_SUPPORT_MODE == 1:
                        for child in children:
                            child.append_other()
                    # Collect children.
                    indvs.extend(children)
                settings.TRANS_MODE = "origin"
                settings.TRANS_CROSS_BAD_INDVS = []
                settings.TRANS_CROSS_BAD_INDVS_HASH = set()
                if generation_timed_out:
                    break
                # The next generation.
                self.population.individuals = indvs

                for analysis_index, a in enumerate(self.analysis):
                    if self._timeout_reached(execution_begin):
                        generation_timed_out = True
                        break
                    if g % a.interval == 0:
                        a.register_step(g=g, population=self.population, engine=self)
                        if self._timeout_reached(execution_begin):
                            if a.__class__.__name__ == "TFISoftInjectionAnalysis":
                                try:
                                    self._run_timeout_repair_grace(analysis_index, g)
                                except Exception:
                                    pass
                            generation_timed_out = True
                            break

                if generation_timed_out:
                    break

                g += 1
        except Exception as e:
            # Log exception info.
            msg = '{} exception is catched'.format(type(e).__name__)
            self.logger.exception(msg)
            raise e
        finally:
            # Perform the analysis post processing.
            for a in self.analysis:
                a.finalize(population=self.population, engine=self)

    def _update_statvars(self):
        '''
        Private helper function to update statistic variables in GA engine, like
        maximum, minimum and mean values.
        '''
        # Wrt original fitness.
        self.ori_fmax = self.population.max(self.ori_fitness)
        self.ori_fmin = self.population.min(self.ori_fitness)
        self.ori_fmean = self.population.mean(self.ori_fitness)

        # Wrt decorated fitness.
        self.fmax = self.population.max(self.fitness)
        self.fmin = self.population.min(self.fitness)
        self.fmean = self.population.mean(self.fitness)

    def _check_parameters(self):
        '''
        Helper function to check parameters of engine.
        '''
        if not isinstance(self.population, Population):
            raise TypeError('population must be a Population object')
        if not isinstance(self.selection, Selection):
            raise TypeError('selection operator must be a Selection instance')
        if not isinstance(self.crossover, Crossover):
            raise TypeError('crossover operator must be a Crossover instance')
        if not isinstance(self.mutation, Mutation):
            raise TypeError('mutation operator must be a Mutation instance')

        for ap in self.analysis:
            if not isinstance(ap, OnTheFlyAnalysis):
                msg = '{} is not subclass of OnTheFlyAnalysis'.format(ap.__name__)
                raise TypeError(msg)

    # Decorators.

    def fitness_register(self, fn):
        '''
        A decorator for fitness function register.
        '''

        @wraps(fn)
        def _fn_with_fitness_check(indv):
            '''
            A wrapper function for fitness function with fitness value check.
            '''
            # Check indv type.
            if not isinstance(indv, Individual):
                raise TypeError('indv\'s class must be Individual or a subclass of Individual')

            # Check fitness.
            fitness = fn(indv)
            is_invalid = (type(fitness) is not float) or (math.isnan(fitness))
            if is_invalid:
                msg = 'Fitness value(value: {}, type: {}) is invalid'
                msg = msg.format(fitness, type(fitness))
                raise ValueError(msg)
            return fitness

        self.fitness = _fn_with_fitness_check
        if self.ori_fitness is None:
            self.ori_fitness = _fn_with_fitness_check

    def analysis_register(self, analysis_cls):
        '''
        A decorator for analysis regsiter.
        '''
        if not issubclass(analysis_cls, OnTheFlyAnalysis):
            raise TypeError('analysis class must be subclass of OnTheFlyAnalysis')

        # Add analysis instance to engine.
        analysis = analysis_cls()
        self.analysis.append(analysis)

    # Functions for fitness scaling.

    def linear_scaling(self, target='max', ksi=0.5):
        '''
        A decorator constructor for fitness function linear scaling.

        :param target: The optimization target, maximization or minimization.
        :type target: str, 'max' or 'min'

        :param ksi: Selective pressure adjustment value.
        :type ksi: float

        Linear Scaling:
            1. arg max f(x), then f' = f - min{f(x)} + ksi;
            2. arg min f(x), then f' = max{f(x)} - f(x) + ksi;
        '''

        def _linear_scaling(fn):
            # For original fitness calculation.
            self.ori_fitness = fn

            @wraps(fn)
            def _fn_with_linear_scaling(indv):
                # Original fitness value.
                f = fn(indv)

                # Determine the value of a and b.
                if target == 'max':
                    f_prime = f - self.ori_fmin + ksi
                elif target == 'min':
                    f_prime = self.ori_fmax - f + ksi
                else:
                    raise ValueError('Invalid target type({})'.format(target))
                return f_prime

            return _fn_with_linear_scaling

        return _linear_scaling

    def dynamic_linear_scaling(self, target='max', ksi0=2, r=0.9):
        '''
        A decorator constructor for fitness dynamic linear scaling.

        :param target: The optimization target, maximization or minimization.
        :type target: str, 'max' or 'min'

        :param ksi0: Initial selective pressure adjustment value, default value
                     is 2
        :type ksi0: float

        :param r: The reduction factor for selective pressure adjustment value,
                  ksi^(k-1)*r is the adjustment value for generation k, default
                  value is 0.9
        :type r: float in range [0.9, 0.999]

        Dynamic Linear Scaling:
            For maximizaiton, f' = f(x) - min{f(x)} + ksi^k, k is generation number.
        '''

        def _dynamic_linear_scaling(fn):
            # For original fitness calculation.
            self.ori_fitness = fn

            @wraps(fn)
            def _fn_with_dynamic_linear_scaling(indv):
                f = fn(indv)
                k = self.current_generation + 1

                if target == 'max':
                    f_prime = f - self.ori_fmin + ksi0 * (r ** k)
                elif target == 'min':
                    f_prime = self.ori_fmax - f + ksi0 * (r ** k)
                else:
                    raise ValueError('Invalid target type({})'.format(target))
                return f_prime

            return _fn_with_dynamic_linear_scaling

        return _dynamic_linear_scaling

    def minimize(self, fn):
        '''
        A decorator for minimizing the fitness function.
        '''

        @wraps(fn)
        def _minimize(indv):
            return -fn(indv)

        return _minimize

