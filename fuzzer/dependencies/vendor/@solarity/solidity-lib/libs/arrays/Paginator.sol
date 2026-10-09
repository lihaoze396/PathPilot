// SPDX-License-Identifier: MIT
pragma solidity ^0.8.0;

import "@openzeppelin/contracts/utils/structs/EnumerableSet.sol";

library Paginator {
    using EnumerableSet for EnumerableSet.AddressSet;

    function part(
        EnumerableSet.AddressSet storage set,
        uint256 offset,
        uint256 limit
    ) internal view returns (address[] memory result) {
        uint256 len = set.length();
        if (offset >= len) {
            return new address[](0);
        }

        uint256 end = offset + limit;
        if (end > len) {
            end = len;
        }

        result = new address[](end - offset);
        for (uint256 i = offset; i < end; i++) {
            result[i - offset] = set.at(i);
        }
    }
}
