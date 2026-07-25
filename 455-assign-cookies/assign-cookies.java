import java.util.Arrays;

class Solution {
    public int findContentChildren(int[] g, int[] s) {

        // Sort greed factors and cookie sizes
        Arrays.sort(g);
        Arrays.sort(s);

        int child = 0;
        int cookie = 0;
        int count = 0;

        // Traverse both arrays
        while (child < g.length && cookie < s.length) {

            // Current cookie can satisfy current child
            if (s[cookie] >= g[child]) {
                count++;
                child++;
                cookie++;
            }
            // Cookie is too small, try a larger cookie
            else {
                cookie++;
            }
        }

        return count;
    }
}