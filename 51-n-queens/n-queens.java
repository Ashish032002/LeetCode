import java.util.*;

class Solution {

    public List<List<String>> solveNQueens(int n) {

        List<List<String>> ans = new ArrayList<>();
        List<String> current = new ArrayList<>();

        solve(0, n, ans, current);

        return ans;
    }

    public void solve(int row, int n,
                      List<List<String>> ans,
                      List<String> current) {
                        
        if (row == n) {
            ans.add(new ArrayList<>(current));
            return;
        }

        for (int i = 0; i < n; i++) {

            if(!isSafe(row, i , current)){
                continue;
            }

            char[] board = new char[n];
            Arrays.fill(board , '.');

            board[i] = 'Q';

            current.add(new String(board));
        
            solve(row + 1, n, ans, current);

            current.remove(current.size() - 1);
        }
    }

    public boolean isSafe(int row, int col,
                           List<String> current) {

        for(int r = 0 ; r < row ; r++){

            String previousRow = current.get(r);

            if(previousRow.charAt(col) == 'Q'){
                return false;
            }

            int queenCol = previousRow.indexOf('Q');

            if(Math.abs(r-row)  == Math.abs(queenCol - col)){
                return false;
            }
        }
        return true;
    }
}