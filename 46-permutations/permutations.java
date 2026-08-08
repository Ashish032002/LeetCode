class Solution {
    public List<List<Integer>> permute(int[] nums) {

        List<List<Integer>> list = new ArrayList<>();
        boolean[] used = new boolean[nums.length];
        List<Integer> current = new ArrayList<>();

        if (nums.length == 0)
            return list;

        solve(nums,current,list,used);

        return list;
    }

public void solve(int[] nums , List<Integer> current ,List<List<Integer>> list , boolean[] used){

    if (current.size() == nums.length) {
        list.add(new ArrayList<>(current));
        return;
    }

    for (int i = 0; i < nums.length; i++) {

        if (used[i]) {
            continue;
        }

        current.add(nums[i]);
        used[i] = true;

        solve(nums, current, list, used);

        current.remove(current.size() - 1);
        used[i] = false;
    }
  }
}