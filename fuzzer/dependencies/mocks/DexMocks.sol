// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract WrappedNativeMock {
    string public constant name = "Wrapped Native";
    string public constant symbol = "WNATIVE";
    uint8 public constant decimals = 18;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    uint256 public totalSupply;

    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Deposit(address indexed account, uint256 amount);
    event Withdrawal(address indexed account, uint256 amount);

    receive() external payable {
        deposit();
    }

    function deposit() public payable {
        balanceOf[msg.sender] += msg.value;
        totalSupply += msg.value;
        emit Deposit(msg.sender, msg.value);
        emit Transfer(address(0), msg.sender, msg.value);
    }

    function withdraw(uint256 amount) external {
        uint256 available = balanceOf[msg.sender];
        uint256 payout = amount <= available ? amount : available;
        balanceOf[msg.sender] = available - payout;
        totalSupply -= payout;
        uint256 cash = payout <= address(this).balance ? payout : address(this).balance;
        if (cash > 0) {
            payable(msg.sender).transfer(cash);
            emit Withdrawal(msg.sender, cash);
            emit Transfer(msg.sender, address(0), cash);
        }
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        _move(msg.sender, to, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        uint256 allowed = allowance[from][msg.sender];
        if (allowed != type(uint256).max) {
            allowance[from][msg.sender] = allowed >= amount ? allowed - amount : 0;
        }
        _move(from, to, amount);
        return true;
    }

    function _move(address from, address to, uint256 amount) private {
        uint256 available = balanceOf[from];
        uint256 moved = amount <= available ? amount : available;
        if (moved > 0) {
            balanceOf[from] = available - moved;
            balanceOf[to] += moved;
        }
        emit Transfer(from, to, moved);
    }
}

contract DexPairMock {
    string public constant name = "Mock LP";
    string public constant symbol = "MLP";
    uint8 public constant decimals = 18;

    mapping(address => uint256) public balanceOf;
    mapping(address => mapping(address => uint256)) public allowance;
    uint256 public totalSupply;

    event Approval(address indexed owner, address indexed spender, uint256 value);
    event Transfer(address indexed from, address indexed to, uint256 value);
    event Burn(address indexed sender, uint256 amount0, uint256 amount1, address indexed to);
    event Mint(address indexed sender, uint256 amount0, uint256 amount1);
    event Swap(
        address indexed sender,
        uint256 amount0In,
        uint256 amount1In,
        uint256 amount0Out,
        uint256 amount1Out,
        address indexed to
    );
    event Sync(uint112 reserve0, uint112 reserve1);

    function token0() external pure returns (address) {
        return 0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c;
    }

    function token1() external pure returns (address) {
        return 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    }

    function getReserves() external view returns (uint112 reserve0, uint112 reserve1, uint32 blockTimestampLast) {
        reserve0 = 1;
        reserve1 = 1;
        blockTimestampLast = uint32(block.timestamp);
    }

    function approve(address spender, uint256 amount) external returns (bool) {
        allowance[msg.sender][spender] = amount;
        emit Approval(msg.sender, spender, amount);
        return true;
    }

    function transfer(address to, uint256 amount) external returns (bool) {
        _move(msg.sender, to, amount);
        return true;
    }

    function transferFrom(address from, address to, uint256 amount) external returns (bool) {
        uint256 allowed = allowance[from][msg.sender];
        if (allowed != type(uint256).max) {
            allowance[from][msg.sender] = allowed >= amount ? allowed - amount : 0;
        }
        _move(from, to, amount);
        return true;
    }

    function mint(address to) external returns (uint256 liquidity) {
        liquidity = 1 ether;
        totalSupply += liquidity;
        balanceOf[to] += liquidity;
        emit Mint(msg.sender, liquidity, liquidity);
        emit Transfer(address(0), to, liquidity);
    }

    function burn(address to) external returns (uint256 amount0, uint256 amount1) {
        amount0 = 1;
        amount1 = 1;
        emit Burn(msg.sender, amount0, amount1, to);
    }

    function skim(address) external {}

    function sync() external {
        emit Sync(1, 1);
    }

    function swap(uint256 amount0Out, uint256 amount1Out, address to, bytes calldata) external {
        emit Swap(msg.sender, 0, 0, amount0Out, amount1Out, to);
    }

    function initialize(address, address) external {}

    function DOMAIN_SEPARATOR() external pure returns (bytes32) {
        return bytes32(0);
    }

    function PERMIT_TYPEHASH() external pure returns (bytes32) {
        return bytes32(0);
    }

    function permit(address, address, uint256, uint256, uint8, bytes32, bytes32) external {}

    function _move(address from, address to, uint256 amount) private {
        uint256 available = balanceOf[from];
        uint256 moved = amount <= available ? amount : available;
        if (moved > 0) {
            balanceOf[from] = available - moved;
            balanceOf[to] += moved;
        }
        emit Transfer(from, to, moved);
    }
}

contract DexFactoryMock {
    address private constant DEFAULT_PAIR = 0x1111111111111111111111111111111111111111;
    address private constant TESTNET_PAIR = 0x2222222222222222222222222222222222222222;

    event PairCreated(address indexed token0, address indexed token1, address pair, uint256 allPairsLength);

    function createPair(address tokenA, address tokenB) external returns (address pair) {
        pair = _pairForFactory();
        emit PairCreated(tokenA, tokenB, pair, 1);
    }

    function getPair(address, address) external view returns (address) {
        return _pairForFactory();
    }

    function allPairs(uint256) external view returns (address) {
        return _pairForFactory();
    }

    function allPairsLength() external pure returns (uint256) {
        return 1;
    }

    function feeTo() external pure returns (address) {
        return address(0);
    }

    function feeToSetter() external pure returns (address) {
        return address(0);
    }

    function _pairForFactory() private view returns (address) {
        if (address(this) == 0x6725F303b657a9451d8BA641348b6761A6CC7a17) {
            return TESTNET_PAIR;
        }
        return DEFAULT_PAIR;
    }
}

contract DexRouterMock {
    address private constant PANCAKE_ROUTER = 0x10ED43C718714eb63d5aA57B78B54704E256024E;
    address private constant UNISWAP_ROUTER = 0x7a250d5630B4cF539739dF2C5dAcb4c659F2488D;
    address private constant PANCAKE_ROUTER_TEST = 0xD99D1c33F9fC3444f8101754aBC46c52416550D1;
    address private constant PANCAKE_ROUTER_TEST_ALT = 0xCc7aDc94F3D80127849D2b41b6439b7CF1eB4Ae0;

    address private constant PANCAKE_FACTORY = 0xcA143Ce32Fe78f1f7019d7d551a6402fC5350c73;
    address private constant UNISWAP_FACTORY = 0x5C69bEe701ef814a2B6a3EDD4B1652CB9cc5aA6f;
    address private constant PANCAKE_FACTORY_TEST = 0x6725F303b657a9451d8BA641348b6761A6CC7a17;

    address private constant WBNB_TOKEN = 0xbb4CdB9CBd36B01bD1cBaEBF2De08d9173bc095c;
    address private constant WETH_TOKEN = 0xC02aaA39b223FE8D0A0e5C4F27eAD9083C756Cc2;
    address private constant WBNB_TEST_TOKEN = 0xae13d989daC2f0dEbFf460aC112a837C89BAa7cd;

    address private constant DEFAULT_PAIR = 0x1111111111111111111111111111111111111111;
    address private constant TESTNET_PAIR = 0x2222222222222222222222222222222222222222;

    receive() external payable {}

    function factory() external view returns (address) {
        address self = address(this);
        if (self == UNISWAP_ROUTER) {
            return UNISWAP_FACTORY;
        }
        if (self == PANCAKE_ROUTER_TEST || self == PANCAKE_ROUTER_TEST_ALT) {
            return PANCAKE_FACTORY_TEST;
        }
        return PANCAKE_FACTORY;
    }

    function WETH() external view returns (address) {
        address self = address(this);
        if (self == UNISWAP_ROUTER) {
            return WETH_TOKEN;
        }
        if (self == PANCAKE_ROUTER_TEST || self == PANCAKE_ROUTER_TEST_ALT) {
            return WBNB_TEST_TOKEN;
        }
        return WBNB_TOKEN;
    }

    function addLiquidity(
        address,
        address,
        uint256 amountADesired,
        uint256 amountBDesired,
        uint256,
        uint256,
        address,
        uint256
    ) external pure returns (uint256 amountA, uint256 amountB, address pair) {
        return (amountADesired, amountBDesired, DEFAULT_PAIR);
    }

    function addLiquidityETH(
        address,
        uint256 amountTokenDesired,
        uint256,
        uint256,
        address,
        uint256
    ) external payable returns (uint256 amountToken, uint256 amountETH, address pair) {
        return (amountTokenDesired, msg.value, _pairForRouter());
    }

    function removeLiquidity(
        address,
        address,
        uint256 liquidity,
        uint256,
        uint256,
        address,
        uint256
    ) external pure returns (uint256 amountA, uint256 amountB) {
        return (liquidity, liquidity);
    }

    function removeLiquidityETH(
        address,
        uint256 liquidity,
        uint256,
        uint256,
        address,
        uint256
    ) external pure returns (uint256 amountToken, uint256 amountETH) {
        return (liquidity, liquidity);
    }

    function removeLiquidityETHSupportingFeeOnTransferTokens(
        address,
        uint256 liquidity,
        uint256,
        uint256,
        address,
        uint256
    ) external pure returns (uint256 amountETH) {
        return liquidity;
    }

    function swapExactETHForTokensSupportingFeeOnTransferTokens(
        uint256,
        address[] calldata,
        address,
        uint256
    ) external payable {}

    function swapExactTokensForETHSupportingFeeOnTransferTokens(
        uint256,
        uint256,
        address[] calldata,
        address,
        uint256
    ) external {}

    function swapExactTokensForTokensSupportingFeeOnTransferTokens(
        uint256,
        uint256,
        address[] calldata,
        address,
        uint256
    ) external {}

    function swapExactETHForTokens(
        uint256,
        address[] calldata path,
        address,
        uint256
    ) external payable returns (uint256[] memory amounts) {
        return _amountVector(path.length, msg.value);
    }

    function swapETHForExactTokens(
        uint256 amountOut,
        address[] calldata path,
        address,
        uint256
    ) external payable returns (uint256[] memory amounts) {
        return _amountVector(path.length, amountOut);
    }

    function swapExactTokensForETH(
        uint256 amountIn,
        uint256,
        address[] calldata path,
        address,
        uint256
    ) external pure returns (uint256[] memory amounts) {
        return _amountVector(path.length, amountIn);
    }

    function swapExactTokensForTokens(
        uint256 amountIn,
        uint256,
        address[] calldata path,
        address,
        uint256
    ) external pure returns (uint256[] memory amounts) {
        return _amountVector(path.length, amountIn);
    }

    function swapTokensForExactETH(
        uint256 amountOut,
        uint256,
        address[] calldata path,
        address,
        uint256
    ) external pure returns (uint256[] memory amounts) {
        return _amountVector(path.length, amountOut);
    }

    function swapTokensForExactTokens(
        uint256 amountOut,
        uint256,
        address[] calldata path,
        address,
        uint256
    ) external pure returns (uint256[] memory amounts) {
        return _amountVector(path.length, amountOut);
    }

    function getAmountsOut(uint256 amountIn, address[] calldata path) external pure returns (uint256[] memory amounts) {
        return _amountVector(path.length, amountIn);
    }

    function getAmountsIn(uint256 amountOut, address[] calldata path) external pure returns (uint256[] memory amounts) {
        return _amountVector(path.length, amountOut);
    }

    function quote(uint256 amountA, uint256 reserveA, uint256 reserveB) external pure returns (uint256 amountB) {
        if (reserveA == 0) {
            return amountA;
        }
        return (amountA * reserveB) / reserveA;
    }

    function _pairForRouter() private view returns (address) {
        address self = address(this);
        if (self == PANCAKE_ROUTER_TEST || self == PANCAKE_ROUTER_TEST_ALT) {
            return TESTNET_PAIR;
        }
        return DEFAULT_PAIR;
    }

    function _amountVector(uint256 pathLength, uint256 seed) private pure returns (uint256[] memory amounts) {
        uint256 size = pathLength > 0 ? pathLength : 2;
        uint256 value = seed > 0 ? seed : 1;
        amounts = new uint256[](size);
        for (uint256 i = 0; i < size; i++) {
            amounts[i] = value;
        }
    }
}
