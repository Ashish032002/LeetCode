class Solution {
    public int uniquePaths(int m, int n) {

    if(m==1 || n==1) return 1;

    int[][] dp = new int[m][n];

    for(int i = 0 ; i<m ; i++){
        dp[i][n-1] = 1;
    }

    for(int j = 0 ; j<n ; j++){
        dp[m-1][j] = 1;
    }

    int row = m-2;
    int col = n-2;

    while(row>=0 && col>=0){
        for(int i = col; i>=0 ; i--){
            dp[row][i] = dp[row+1][i] + dp[row][i+1];
        }

        for(int j = row; j>=0 ; j--){
            dp[j][col] = dp[j][col + 1] + dp[j+1][col];
        }
        row--;
        col--;
    }

    return dp[0][0];   
    }
}
