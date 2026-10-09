#!/usr/bin/env python3
# -*- coding: utf-8 -*-

class FuzzingEnvironment:
    def __init__(self, **kwargs) -> None:
        self.nr_of_transactions = 0
        self.setup_transactions = 0
        self.current_generation = 0
        self.unique_individuals = set()
        self.code_coverage = set()
        self.children_code_coverage = dict()
        self.previous_code_coverage_length = 0
        self.branch_coverage = set()
        self.function_coverage = set()
        self.function_coverage_names = set()

        self.visited_branches = dict()

        self.memoized_fitness = dict()
        self.memoized_storage = dict()
        self.memoized_symbolic_execution = dict()

        self.individual_branches = dict()

        self.data_dependencies = dict()

        self.__dict__.update(kwargs)
