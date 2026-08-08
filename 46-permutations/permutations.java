class Solution {
    public List<List<Integer>> permute(int[] nums) {
     
     List<List<Integer>> list = new ArrayList<>();
     List<Integer> current = new ArrayList<>();
     boolean[] used = new boolean[nums.length];

     solve(nums , list , current , used);

     if (nums.length == 0)
            return list;
     
     return list;
  }

  public void solve(int[] nums , List<List<Integer>> list , List<Integer> current , boolean[] used){

    if(current.size() == nums.length){
        list.add(new ArrayList<>(current));
            return;
    }

    for(int i = 0 ; i < nums.length ; i++){

        if(used[i]){
            continue;
        }
        current.add(nums[i]);
        used[i] = true;

        solve(nums , list , current , used);

        current.remove(current.size() - 1);
            used[i] = false;


    }

  }
}