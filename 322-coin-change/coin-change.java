class Solution {
    public int coinChange(int[] coins, int amount) {
        int INF = (int)1e9;

        int[] dp = new int[amount + 1];
        Arrays.fill(dp, INF);
        dp[0] = 0;

        for (int coin : coins) {
            for (int amt = coin; amt <= amount; amt++) {
                dp[amt] = Math.min(dp[amt], 1 + dp[amt - coin]);
            }
        }

        return dp[amount] >= INF ? -1 : dp[amount];
    }
}