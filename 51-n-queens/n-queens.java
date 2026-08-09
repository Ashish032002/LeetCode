import java.util.*;

class Solution {

    private int n;
    private List<List<String>> ans;

    public List<List<String>> solveNQueens(int n) {

        this.n = n;
        this.ans = new ArrayList<>();

        char[][] board = new char[n][n];

        for (int i = 0; i < n; i++) {
            Arrays.fill(board[i], '.');
        }

        backtrack(0, 0, 0, 0, board);

        return ans;
    }

    private void backtrack(int row,
                           int cols,
                           int diag1,
                           int diag2,
                           char[][] board) {

        // All queens placed
        if (row == n) {

            List<String> solution = new ArrayList<>();

            for (int i = 0; i < n; i++) {
                solution.add(new String(board[i]));
            }

            ans.add(solution);
            return;
        }

     
        int available =
                ((1 << n) - 1) & ~(cols | diag1 | diag2);

        while (available != 0) {

       
            int position = available & -available;

            available -= position;

          
            int col = Integer.numberOfTrailingZeros(position);

            board[row][col] = 'Q';

            backtrack(
                row + 1,
                cols | position,
                (diag1 | position) << 1,
                (diag2 | position) >> 1,
                board
            );

           
            board[row][col] = '.';
        }
    }
}