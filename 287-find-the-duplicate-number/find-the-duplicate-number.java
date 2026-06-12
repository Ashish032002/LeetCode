class Solution {
    public int findDuplicate(int[] nums) {
        int n = nums.length - 1;
        int duplicate = 0;

        int maxBit = 31 - Integer.numberOfLeadingZeros(n);

        for (int bit = 0; bit <= maxBit; bit++) {
            int mask = 1 << bit;

            int numsCount = 0;
            int rangeCount = 0;

            for (int num : nums) {
                if ((num & mask) != 0) {
                    numsCount++;
                }
            }

            for (int i = 1; i <= n; i++) {
                if ((i & mask) != 0) {
                    rangeCount++;
                }
            }

            if (numsCount > rangeCount) {
                duplicate |= mask;
            }
        }

        return duplicate;
    }
}