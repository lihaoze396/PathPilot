// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract DefiProxyMock {
    uint256 public calls;

    receive() external payable {}

    fallback() external payable {
        calls += 1;
    }
}
