class Solution {
    public int[] gardenNoAdj(int n, int[][] paths) {

        int[] color = new int[n];

     
        for (int garden = 0; garden < n; garden++) {

            boolean[] used = new boolean[5];

        
            for (int[] path : paths) {

                int u = path[0] - 1;
                int v = path[1] - 1;

                if (u == garden) {
                    used[color[v]] = true;
                }

                if (v == garden) {
                    used[color[u]] = true;
                }
            }

            for (int flower = 1; flower <= 4; flower++) {
                if (!used[flower]) {
                    color[garden] = flower;
                    break;
                }
            }
        }

        return color;
    }
}