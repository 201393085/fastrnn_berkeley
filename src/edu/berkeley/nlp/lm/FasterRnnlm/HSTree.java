package edu.berkeley.nlp.lm.FasterRnnlm;

import com.sun.javafx.binding.StringFormatter;

import java.io.DataInputStream;
import java.io.IOException;
import java.util.logging.Logger;

import static edu.berkeley.nlp.lm.FasterRnnlm.Constant.MAX_HSTREE_DEPTH;
import static java.lang.Math.exp;
import static java.lang.Math.log;
import static java.lang.Math.max;
import static java.lang.Math.abs;

/**
 * Created by nlp on 16-12-6.
 */
public class HSTree {
    private static String strClassName = HSTree.class.getName();
    private static Logger logger = Logger.getLogger(strClassName);

    private class Tree{
        int leaf_count_;
        int arity_;
        int root_node_;
        int tree_height_;

        int[] children_;
        int[] path_lengths_;
        int[] points_;
        int[] branches_;

        public Tree(int leaf_count, int[] children, int arity){
            this.leaf_count_ = leaf_count;
            this.arity_ = arity;
            this.root_node_ = -1;
            this.tree_height_ = -1;
            this.children_ = children;
            this.path_lengths_ = new int[leaf_count];
            this.points_ = new int[leaf_count * (MAX_HSTREE_DEPTH + 1)];
            this.branches_ = new int[leaf_count * MAX_HSTREE_DEPTH];

            int extra_node_count = (leaf_count - 1) / (arity_ - 1);
            int node_count = leaf_count + extra_node_count;
            int[] branch_id = new int[node_count];
            int[] parent_node = new int[node_count];

            if ((leaf_count - 1) % (arity_ - 1) != 0) {
                logger.severe(String.format("Cannot build a full tree of arity %d from %d leaf nodes\n",arity_,leaf_count));
                return;
            }

            root_node_ = node_count - 1;

            // build parents by children
            for (int parent = 0; parent < extra_node_count; parent++) {
                for (int branch = 0; branch < arity_; ++branch) {
                    int child_index = children_[parent * arity_ + branch];
                    if (child_index >= node_count) {
                        logger.severe(String.format("ERROR bad child index (%d)\n", child_index));
                        return;
                    }
                    parent_node[child_index] = leaf_count + parent;
                    branch_id[child_index] = branch;
                }
            }

            // Now assign branch_id code to each vocabulary word
            for (int leaf_node = 0; leaf_node < leaf_count; leaf_node++) {
                int[] path_nodes = new int[MAX_HSTREE_DEPTH];
                int[] path_branches = new int[MAX_HSTREE_DEPTH];
                int path_length = 0;
                for (int node = leaf_node; node != root_node_; node = parent_node[node]) {
                    if (path_length == MAX_HSTREE_DEPTH) {
                        logger.severe(String.format("ERROR Cannot build a tree with height greater than %d\n", MAX_HSTREE_DEPTH));
                        return;
                    }
                    path_branches[path_length] = branch_id[node];
                    path_nodes[path_length] = node;
                    path_length++;
                }

                path_lengths_[leaf_node] = path_length + 1;

                int path_nodes_offset = leaf_node * (MAX_HSTREE_DEPTH + 1);
                int path_branches_offset = leaf_node * MAX_HSTREE_DEPTH;
                points_[path_nodes_offset] = root_node_;
                for (int i = 0; i < path_length; i++) {
                    branches_[path_branches_offset+path_length - i - 1] = path_branches[i];
                    points_[path_nodes_offset+path_length - i] = path_nodes[i];
                }
            }

            tree_height_ = 0;
            for (int i = 0; i < leaf_count_; i++) {
                tree_height_ = max(tree_height_, GetPathLength(i) - 1);
            }
        }

        int GetPathLength(int word) { return path_lengths_[word]; }

        int GetChildOffset(int node, int branch) {
            return (node - leaf_count_) * (arity_ - 1) + branch;
        }
    }

    private int layer_size;
    private int syn_size;
    private Matrix weights_;
    private Tree tree_;

    public HSTree(int vocab_size, int layer_size, int arity, int[] children){
        this.layer_size = layer_size;
        this.syn_size = layer_size*vocab_size;
        this.weights_ = new Matrix(vocab_size, layer_size);
        this.tree_ = new Tree(vocab_size, children, arity);
    }

    public void Load(DataInputStream dis) throws IOException {
        weights_.ReadMatrix(dis);
    }

    public static HSTree CreateHuffmanTree(Vocabulary vocab, int layer_size, int arity) {
        int extra_node_count = (vocab.size() - 1) / (arity - 1);
        int node_count = vocab.size() + extra_node_count;
        long[] weight = new long[node_count + 2];
        int[] children = new int[extra_node_count * arity];

        for (int i = 0; i < vocab.size(); i++) {
            weight[i] = vocab.GetFreqByIndex(i);
        }
        for (int i = vocab.size(); i < node_count; i++) {
            weight[i] = Long.MAX_VALUE;
        }

        int next_leaf_node = vocab.size() - 1;
        int next_inner_node = vocab.size();

        int[] min_indices = new int[arity];
        for (int new_node = vocab.size(); new_node < node_count; new_node++) {
            // First, find exactly arity smallest nodes
            // and store their indices in min_indices[new_node]
            for (int branch = 0; branch < arity; ++branch) {
                if (next_leaf_node >= 0 && weight[next_leaf_node] < weight[next_inner_node]) {
                    min_indices[branch] = next_leaf_node;
                    next_leaf_node--;
                } else {
                    min_indices[branch] = next_inner_node;
                    next_inner_node++;
                }
            }

            // Then, build a new node
            weight[new_node] = 0;
            for (int branch = 0; branch < arity; ++branch) {
                int child_index = min_indices[branch];
                weight[new_node] += weight[child_index];
                children[(new_node - vocab.size()) * arity + branch] = child_index;
            }
        }
        return new HSTree(vocab.size(), layer_size, arity, children);
    }

    void CalculateNodeChildrenScores(
        HSTree hs, int node, double[] hidden,
        long[] feature_hashes, int maxent_order, MaxEnt maxent,
        double[] branch_scores
    ) {
        int arity = hs.tree_.arity_;
        for (int branch = 0; branch < arity - 1; ++branch) {
            branch_scores[branch] = 0;
            int child_offset = hs.tree_.GetChildOffset(node, branch);
            double[] sm_embedding = hs.weights_.data[child_offset];
            for (int i = 0; i < hs.layer_size; ++i) {
                branch_scores[branch] += hidden[i] * sm_embedding[i];
            }
            for (int order = 0; order < maxent_order; ++order) {
                long maxent_index = feature_hashes[order] + child_offset;
                branch_scores[branch] += maxent.GetValue(maxent_index);
            }
        }
    }

    void PropagateNodeForward(
        HSTree hs, int node, double[] hidden,
        long[] feature_hashes, int maxent_order, MaxEnt maxent,
        double[] state
    ) {
        int arity = hs.tree_.arity_;
        double[] tmp = new double[arity];
        CalculateNodeChildrenScores(hs, node, hidden, feature_hashes, maxent_order, maxent, tmp);

        double max_score = 0;
        for (int i = 0; i < arity - 1; ++i) {
            max_score = (tmp[i] > max_score) ? tmp[i] : max_score;
        }
        state[arity - 1] = exp(-max_score);

        double f = state[arity - 1];
        for (int i = 0; i < arity - 1; ++i) {
            state[i] = exp(tmp[i] - max_score);
            f += state[i];
        }
        for (int i = 0; i < arity; ++i) {
            state[i] /= f;
        }
    }


    public double CalculateLog10Probability(
            int target_word, long[] feature_hashes, int maxent_order,
            boolean dynamic_maxent_prunning, double[] hidden, MaxEnt maxent
    ) {
        int arity = tree_.arity_;
        double[] softmax_state = new double[arity];
        double logprob = 0.0;
        if(target_word == tree_.path_lengths_.length) return 0.0;
        for (int depth = 0; depth < tree_.GetPathLength(target_word) - 1; depth++) {
            int node = tree_.points_[target_word*(MAX_HSTREE_DEPTH + 1)+depth];

            if (dynamic_maxent_prunning) {
                for (int order = 0; order < maxent_order; ++order) {
                    for (int branch = 0; branch < arity - 1; ++branch) {
                        int child_offset = tree_.GetChildOffset(node, branch);
                        long maxent_index = feature_hashes[order] + child_offset;
                        if (abs(maxent_index)<0.00000001) {
                            maxent_order = order;
                            break;
                        }
                    }
                }
            }

            PropagateNodeForward(
                    this, node, hidden,
                    feature_hashes, maxent_order, maxent,
                    softmax_state);

        int selected_branch = tree_.branches_[target_word*MAX_HSTREE_DEPTH + depth];
            logprob += log(softmax_state[selected_branch]);
        }

        return logprob;
    }

    public static void main(String[] args) {
        System.out.println(Long.MAX_VALUE);
    }
}
