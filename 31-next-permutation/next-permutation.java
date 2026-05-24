class Solution {

    public void nextPermutation(int[] nums) {

        int idx = -1;

        for(int i = nums.length - 2; i >= 0; i--) {
            if(nums[i] < nums[i + 1]) {
                idx = i;
                break;
            }
        }

        if(idx == -1) {
            reverse(nums, 0, nums.length - 1);
            return;
        }

        for(int i = nums.length - 1; i > idx; i--) {
            if(nums[i] > nums[idx]) {
                swap(nums, i, idx);
                break;
            }
        }

        reverse(nums, idx + 1, nums.length - 1);
    }

    public void swap(int[] nums, int i, int j) {
        int temp = nums[i];
        nums[i] = nums[j];
        nums[j] = temp;
    }

    public void reverse(int[] nums, int index, int lastIndex) {
        while(index < lastIndex) {
            swap(nums, index, lastIndex);
            index++;
            lastIndex--;
        }
    }
}