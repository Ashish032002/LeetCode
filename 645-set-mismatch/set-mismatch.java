class Solution {
    public int[] findErrorNums(int[] nums) {
    int n = nums.length;
    long Sn = (long) n * (n + 1) / 2;
    long S2n = (long) n * (n + 1) * (2 * n + 1) / 6;
    long S = 0; 
    long S2 = 0;

    for(int i= 0 ;  i < nums.length ; i++){
        S += nums[i];
        S2 += (long)nums[i]*nums[i];
    }

    long val1 = S-Sn;
    long val2 = S2-S2n;
    val2 = val2/val1;
    long x = (val1 + val2) / 2;
    long y = x - val1;

    return new int[]{(int) x, (int) y};
    }   
}