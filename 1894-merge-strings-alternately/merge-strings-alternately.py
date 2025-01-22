class Solution(object):
    def mergeAlternately(self, word1, word2):
        result = []
        i = 0

        while len(word1)>i or len(word2)>i:
            if len(word1)>i:
                result.append(word1[i])
            if len(word2)>i:
                result.append(word2[i])

            i+=1
        return ''.join(result)            


            


            