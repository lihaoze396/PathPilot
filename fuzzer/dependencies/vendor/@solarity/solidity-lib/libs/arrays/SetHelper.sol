// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

library SetHelper {
    using EnumerableSet for EnumerableSet.AddressSet;

    function add(EnumerableSet.AddressSet storage set, address[] calldata values) internal {
        for (uint256 i = 0; i < values.length; i++) {
            set.add(values[i]);
        }
    }

    function remove(EnumerableSet.AddressSet storage set, address[] calldata values) internal {
        for (uint256 i = 0; i < values.length; i++) {
            set.remove(values[i]);
        }
    }
}
