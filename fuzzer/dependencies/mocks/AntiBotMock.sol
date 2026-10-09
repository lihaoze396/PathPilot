// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

contract AntiBotMock {
    function protect(address, address, uint256) external pure returns (uint256 fee) {
        return 0;
    }
}
