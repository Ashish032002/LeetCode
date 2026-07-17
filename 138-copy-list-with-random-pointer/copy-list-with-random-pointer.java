/*
// Definition for a Node.
class Node {
    int val;
    Node next;
    Node random;

    public Node(int val) {
        this.val = val;
        this.next = null;
        this.random = null;
    }
}
*/

class Solution {
    public Node copyRandomList(Node head) {

       
        if (head == null) {
            return head;
        }

        // Step 1: Weave copied nodes into the original list
        Node curr = head;

        while (curr != null) {
            Node copy = new Node(curr.val);

            copy.next = curr.next;
            curr.next = copy;

            curr = copy.next;
        }

        // Step 2: Assign random pointers
        curr = head;

        while (curr != null) {
            if (curr.random != null) {
                curr.next.random = curr.random.next;
            }

            curr = curr.next.next;
        }

        // Step 3: Separate the original and copied lists
        curr = head;
        Node copyHead = head.next;

        while (curr != null) {
            Node copy = curr.next;

            // Restore original list
            curr.next = copy.next;

            // Connect copied list
            if (copy.next != null) {
                copy.next = copy.next.next;
            }

            curr = curr.next;
        }

        return copyHead;
    }
}