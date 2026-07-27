import java.util.*;

class Solution {

    public List<List<Integer>> subsetsWithDup(int[] nums) {

        Arrays.sort(nums); // Important for handling duplicates

        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> current = new ArrayList<>();

        solve(0, nums, current, ans);

        return ans;
    }

    public void solve(int index, int[] nums,
                      List<Integer> current,
                      List<List<Integer>> ans) {

        // Every recursive call represents one valid subset
        ans.add(new ArrayList<>(current));

        for (int i = index; i < nums.length; i++) {

            // Skip duplicate elements at the same level
            if (i > index && nums[i] == nums[i - 1]) {
                continue;
            }

            // Take
            current.add(nums[i]);

            solve(i + 1, nums, current, ans);

            // Backtrack
            current.remove(current.size() - 1);
        }
    }
}