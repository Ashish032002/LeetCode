import java.util.Arrays;

class Solution {
    public int[] findErrorNums(int[] nums) {

        Arrays.sort(nums);

        int duplicate = -1;
        int missing = -1;

        // Find duplicate
        for (int i = 0; i < nums.length - 1; i++) {
            if (nums[i] == nums[i + 1]) {
                duplicate = nums[i];
                break;
            }
        }

        // Find missing
        if (nums[0] != 1) {
            missing = 1;
        } else {
            for (int i = 0; i < nums.length - 1; i++) {

                if (nums[i] == nums[i + 1]) {
                    continue;
                }

                if (nums[i + 1] != nums[i] + 1) {
                    missing = nums[i] + 1;
                    break;
                }
            }

            if (missing == -1) {
                missing = nums[nums.length - 1] + 1;
            }
        }

        return new int[]{duplicate, missing};
    }
}