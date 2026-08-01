class Solution {
    public List<List<Integer>> combinationSum(int[] candidates, int target) {

        List<List<Integer>> ans = new ArrayList<>();
        List<Integer> current = new ArrayList<>();

        solve( 0, candidates , target , 0 , current , ans );

        return ans;
        
    }

    public void solve(int index, int[] candidates, int target, int currentSum, List<Integer> current,
        List<List<Integer>> ans)
    {

        if (currentSum == target) {
            ans.add(new ArrayList<>(current));
            return;
        }

       
        if (currentSum > target) {
            return;
        }

        for(int i = index ; i < candidates.length ; i ++){

            current.add(candidates[i]);

            solve(i , candidates,target,currentSum + candidates[i],current, ans);

            current.remove(current.size() - 1);

        }

    }
}