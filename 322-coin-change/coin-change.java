class Solution {
    public int coinChange(int[] coins, int amount) {
       if(amount< 1) return 0;

       int[] minCoinDP = new int[amount+1];

       for(int i= 1; i<=amount ; i++)
       {
        minCoinDP[i] = Integer.MAX_VALUE;

        for(int coin:coins)
        {
            if(coin<=i && minCoinDP[i-coin] != Integer.MAX_VALUE)
            {
                minCoinDP[i]= Math.min(minCoinDP[i],1+minCoinDP[i-coin]);
            }
        }

       }
       if(minCoinDP[amount] == Integer.MAX_VALUE)
       {
           return -1;
       }

       return minCoinDP[amount];
    }
}