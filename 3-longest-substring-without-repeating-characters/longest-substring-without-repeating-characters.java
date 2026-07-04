class Solution {
    public int lengthOfLongestSubstring(String s) {

        HashMap<Character, Integer> map = new HashMap<>();
        int left = 0 ;
        int maxlength = 0;

        for(int right = 0 ; right < s.length() ; right++){
            char current = s.charAt(right);

            if(map.containsKey(current) && map.get(current) >= left){
                left = map.get(current) + 1;
            }

            map.put(current , right);

            maxlength = Math.max(maxlength , right - left + 1);
        } 

        return maxlength;      
    }
}