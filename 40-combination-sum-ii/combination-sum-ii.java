class Solution {
    public List<List<Integer>> combinationSum2(int[] candidates, int target) {

        Arrays.sort(candidates);

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

            if (i > index && candidates[i] == candidates[i - 1]) {
                continue;
            }

            current.add(candidates[i]);

            solve(i + 1 , candidates,target,currentSum + candidates[i],current, ans);

            current.remove(current.size() - 1);

        }

    }
}