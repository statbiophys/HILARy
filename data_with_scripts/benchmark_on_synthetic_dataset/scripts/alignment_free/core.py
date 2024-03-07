import airr
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

W_l = 150
per = 0.1


def alignement_free_clone(repertoire, table_neg, W_l=150, per=0.1):
    MM = repertoire.shape[0]
    df = pd.DataFrame(repertoire.SEQUENCE[:].values)
    df2 = pd.DataFrame(table_neg.values)
    table_r_f = df.append(df2, ignore_index=True)
    table_last_for_neg = truncate_sequence_rf(table_r_f[0], W_l, 0)
    vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, use_idf=True)
    tf_idf_matrix_full_for_neg = vectorizer.fit_transform(table_last_for_neg.end)
    matches_fast_neg = awesome_cossim_top(
        tf_idf_matrix_full_for_neg[:MM, :],
        tf_idf_matrix_full_for_neg[MM:, :].transpose(),
        5,
        0,
    )
    dist2nearestcosine_neg = compute_dist2nearest(matches_fast_neg)

    thresh_cosine = np.percentile(dist2nearestcosine_neg, per)
    table_last = truncate_sequence_v(repertoire.SEQUENCE, W_l)

    tf_idf_matrix_full = vectorizer.fit_transform(table_last.end)
    matches_fast_s = awesome_cossim_top(
        tf_idf_matrix_full, tf_idf_matrix_full.transpose(), 1800, 0.5
    )
    dist2nearestcosine = compute_dist2nearest(matches_fast_s)

    clusters_cosine_full_s = assign_clones(matches_fast_s, thresh_cosine)
    table_out = pd.DataFrame(columns=["ID", "SEQUENCE", "CLONE"])
    table_out.SEQUENCE = repertoire.SEQUENCE
    table_out.ID = repertoire.values[:, 0]
    table_out.CLONE = clusters_cosine_full_s
    airr.dump_rearrangement(table_out, "output_table.tsv")
    return table_out


def specificity_comp(table_clone_list, clusters_cosine_full, M):
    sum_pairs = 0
    sum_clustered_pairs = 0
    for l in range(0, max(table_clone_list)):
        temp_in = np.where(table_clone_list == l)[0]
        temp_out = np.where(table_clone_list != l)[0]
        temp_out = np.random.choice(temp_out, M)
        list_comb = list(itertools.product(temp_in, temp_out))
        per_clone_score = 0
        sum_pairs = sum_pairs + list_comb.__len__()
        for i, j in list_comb:

            per_clone_score = per_clone_score + (
                clusters_cosine_full[i] != clusters_cosine_full[j]
            )
        sum_clustered_pairs = sum_clustered_pairs + per_clone_score
    specificity = sum_clustered_pairs / sum_pairs
    return specificity


def plot_mean_and_CI(
    x_vec, mean, lb, ub, color_mean=None, color_shading=None, name1="b"
):
    # plt.set_facecolor('w')
    # plot the shaded range of the confidence intervals
    plt.style.use("_classic_test")
    plt.grid(None)
    plt.fill_between(x_vec, ub, lb, color=color_shading, alpha=0.3)

    # plot the mean on top
    plt.plot(x_vec, mean, color_mean, label=name1, linewidth=10)


def PPV_comp_ig(table_clone_list, clusters_cosine_full):
    # M_L=2000000
    sum_pairs = 0
    sum_clustered_pairs = 0
    for i in range(0, max(clusters_cosine_full)):
        temp_in = np.where(clusters_cosine_full == i)
        if temp_in[0].shape[0] > 1:
            combinations = list(itertools.combinations(temp_in[0], 2))
            sum_pairs = sum_pairs + combinations.__len__()
            per_clone_score = 0
            for j in range(0, combinations.__len__()):
                per_clone_score = per_clone_score + (
                    table_clone_list[combinations[j][0]]
                    == table_clone_list[combinations[j][1]]
                )
            sum_clustered_pairs = sum_clustered_pairs + per_clone_score
    PPV = sum_clustered_pairs / sum_pairs
    return PPV


def ngrams(string, n=5):
    string = re.sub(r"[,-./]|\sBD", r"", string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return ["".join(ngram) for ngram in ngrams]


vectorizer = TfidfVectorizer(min_df=1, analyzer=ngrams, use_idf=True)


def truncate_sequence_v(table, W_l=200, E_l=0):
    import pandas as pd

    table_last = pd.DataFrame(columns=["end"])
    sequencesplit = []
    M = table.shape[0]

    # W_l=200

    for i in range(0, M):
        s_l = table[i].__len__()
        table_last = table_last.append(
            {"end": table[i][s_l - W_l : s_l - E_l]}, ignore_index=True
        )
    return table_last


def sift4(s1, s2, max_offset=5):
    """
    This is an implementation of general Sift4.
    """
    t1, t2 = list(s1), list(s2)
    l1, l2 = len(t1), len(t2)

    if not s1:
        return l2

    if not s2:
        return l1

    # Cursors for each string
    c1, c2 = 0, 0

    # Largest common subsequence
    lcss = 0

    # Local common substring
    local_cs = 0

    # Number of transpositions ('ab' vs 'ba')
    trans = 0

    # Offset pair array, for computing the transpositions
    offsets = []

    while c1 < l1 and c2 < l2:
        if t1[c1] == t2[c2]:
            local_cs += 1

            # Check if current match is a transposition
            is_trans = False
            i = 0
            while i < len(offsets):
                ofs = offsets[i]
                if c1 <= ofs["c1"] or c2 <= ofs["c2"]:
                    is_trans = abs(c2 - c1) >= abs(ofs["c2"] - ofs["c1"])
                    if is_trans:
                        trans += 1
                    elif not ofs["trans"]:
                        ofs["trans"] = True
                        trans += 1
                    break
                elif c1 > ofs["c2"] and c2 > ofs["c1"]:
                    del offsets[i]
                else:
                    i += 1
            offsets.append({"c1": c1, "c2": c2, "trans": is_trans})

        else:
            lcss += local_cs
            local_cs = 0
            if c1 != c2:
                c1 = c2 = min(c1, c2)

            for i in range(max_offset):
                if c1 + i >= l1 and c2 + i >= l2:
                    break
                elif c1 + i < l1 and s1[c1 + i] == s2[c2]:
                    c1 += i - 1
                    c2 -= 1
                    break

                elif c2 + i < l2 and s1[c1] == s2[c2 + i]:
                    c2 += i - 1
                    c1 -= 1
                    break

        c1 += 1
        c2 += 1

        if c1 >= l1 or c2 >= l2:
            lcss += local_cs
            local_cs = 0
            c1 = c2 = min(c1, c2)

    lcss += local_cs
    return round(max(l1, l2) - lcss + trans)


def error_vj_compute(clusters_cosine_full_short, V_cell, J_cell):
    M_L = 2000000
    error_J = 0
    error_V = 0
    error_loc_v = []
    error_loc_j = []
    for i in range(0, M_L):
        a = np.where(clusters_cosine_full_short == i)[0]
        if a.size > 1:
            size_clust_v = np.unique(V_cell[2][a]).size - 1
            error_V = error_V + size_clust_v
            size_clust_j = np.unique(J_cell[2][a]).size - 1
            error_J = error_J + size_clust_j

            if size_clust_v > 0:
                error_loc_v.append(a)
            if size_clust_j > 0:
                error_loc_j.append(a)
    return error_V, error_J, error_loc_v, error_loc_j


def compute_cluster_stats(clusters_cosine_full, table, M):
    M_points = M
    clustersizevec_cosine = []
    for i in range(0, max(clusters_cosine_full)):
        temp = np.where(clusters_cosine_full == i)
        if temp[0].shape[0] > 1:
            clustersizevec_cosine = np.append(clustersizevec_cosine, temp[0].shape[0])

    clustersizevec_table = []
    for i in range(0, max(table.CLONE[0:M_points])):
        temp = np.where(table.CLONE[0:M_points] == i)
        if temp[0].shape[0] > 1:
            clustersizevec_table = np.append(clustersizevec_table, temp[0].shape[0])
    junction_length = np.zeros(table.shape[0])
    for i in range(0, table.shape[0]):
        junction_length[i] = table.JUNCTION[i].__len__()

    junctionsizevec_cosine = []
    for i in range(0, max(clusters_cosine_full)):
        temp = np.where(clusters_cosine_full == i)
        if temp[0].shape[0] > 1:
            junctionsizevec_cosine = np.append(
                junctionsizevec_cosine, np.unique(junction_length[temp]).shape[0]
            )
    junctionsizevec_table = []
    for i in range(0, int(max(table.CLONE[0:M_points]))):
        temp = np.where(table.CLONE[0:M_points] == i)
        if temp[0].shape[0] > 1:
            junctionsizevec_table = np.append(
                junctionsizevec_table, np.unique(junction_length[temp[0]]).shape[0]
            )
    return (
        clustersizevec_cosine,
        clustersizevec_table,
        junctionsizevec_cosine,
        junctionsizevec_table,
    )


def vj_cell_extract(table, M):
    table_v = pd.DataFrame(columns=["V"])
    table_j = pd.DataFrame(columns=["J"])
    for i in range(0, table.__len__()):
        table_v = table_v.append({"V": table.V_CALL[i][0:15]}, ignore_index=True)
        table_j = table_j.append({"J": table.J_CALL[i][0:13]}, ignore_index=True)
    J_cell = np.unique(
        table_j.J, return_index=True, return_inverse=True, return_counts=True
    )
    V_cell = np.unique(
        table_v.V, return_index=True, return_inverse=True, return_counts=True
    )
    table_vj = pd.DataFrame(columns=["VJ"])
    for i in range(0, M):
        temp = table_v.V[i] + table_j.J[i]
        table_vj = table_vj.append({"VJ": temp}, ignore_index=True)
    VJ_cell = np.unique(
        table_vj.VJ, return_index=True, return_inverse=True, return_counts=True
    )
    return V_cell, J_cell, VJ_cell


def get_csrt_ntop(csr_row, ntop):
    nnz = csr_row.getnnz()
    if nnz == 0:
        return None
    elif nnz <= ntop:
        resultt = zip(gt_csr_row.indices, csr_row.data)
    else:
        arg_idx = np.argpartition(csr_row.data, -ntop)[-ntop:]
        result = zip(csr_row.indices[arg_idx], csr_row.data[arg_idx])
    return sorted(result, key=lambda x: -x[1])


def scipy_cossim_top(A, B, ntop, lower, bound=0):
    C = A.dot(B)
    return [get_csrt_ntop(row, ntop) for row in C]


def listify(obj, length):
    if type(obj) is not list:
        obj = [obj] * length
    elif len(obj) > length:
        raise ValueError("{} must be of length {}".format(obj, length))
    elif len(obj) < length:
        if obj[0] is None:
            # append nones to start
            obj = [None] * (length - len(obj)) + obj
        else:
            # append nones to end
            obj = obj + [None] * (length - len(obj))
    return obj


import re


def ngrams_old(string, n=4):
    string = re.sub(r"[,-./]|\sBD", r"", string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return ["".join(ngram) for ngram in ngrams]


def ngrams(string, n=5):
    string = re.sub(r"[,-./]|\sBD", r"", string)
    ngrams = zip(*[string[i:] for i in range(n)])
    return ["".join(ngram) for ngram in ngrams]


def get_csrt_ntop(csr_row, ntop):
    nnz = csr_row.getnnz()
    if nnz == 0:
        return None
    elif nnz <= ntop:
        resultt = zip(gt_csr_row.indices, csr_row.data)
    else:
        arg_idx = np.argpartition(csr_row.data, -ntop)[-ntop:]
        result = zip(csr_row.indices[arg_idx], csr_row.data[arg_idx])
    return sorted(result, key=lambda x: -x[1])


def scipy_cossim_top(A, B, ntop, lower, bound=0):
    C = A.dot(B)
    return [get_csrt_ntop(row, ntop) for row in C]


import numpy as np
import sparse_dot_topn.sparse_dot_topn as ct
from scipy.sparse import csr_matrix


def awesome_cossim_top(A, B, ntop, lower_bound=0):
    # force A and B as a CSR matrix.
    # If they have already been CSR, there is no overhead
    A = A.tocsr()
    B = B.tocsr()
    M, _ = A.shape
    _, N = B.shape

    idx_dtype = np.int32

    nnz_max = M * ntop

    indptr = np.zeros(M + 1, dtype=idx_dtype)
    indices = np.zeros(nnz_max, dtype=idx_dtype)
    data = np.zeros(nnz_max, dtype=A.dtype)

    ct.sparse_dot_topn(
        M,
        N,
        np.asarray(A.indptr, dtype=idx_dtype),
        np.asarray(A.indices, dtype=idx_dtype),
        A.data,
        np.asarray(B.indptr, dtype=idx_dtype),
        np.asarray(B.indices, dtype=idx_dtype),
        B.data,
        ntop,
        lower_bound,
        indptr,
        indices,
        data,
    )

    return csr_matrix((data, indices, indptr), shape=(M, N))


def get_matches_df(sparse_matrix, name_vector, top=100):
    non_zeros = sparse_matrix.nonzero()

    sparserows = non_zeros[0]
    sparsecols = non_zeros[1]

    if top:
        nr_matches = top
    else:
        nr_matches = sparsecols.size

    left_side = np.empty([nr_matches], dtype=object)
    right_side = np.empty([nr_matches], dtype=object)
    similairity = np.zeros(nr_matches)

    for index in range(0, nr_matches):
        left_side[index] = name_vector[sparserows[index]]
        right_side[index] = name_vector[sparsecols[index]]
        similairity[index] = sparse_matrix.data[index]

    return pd.DataFrame(
        {"left_side": left_side, "right_side": right_side, "similairity": similairity}
    )


import itertools


def sort_coo(csr_row):

    result = zip(csr_row.indices, csr_row.data)
    return sorted(result, key=lambda x: -x[1])


def truncate_sequence(table, W_l=200):
    import pandas as pd

    table_last = pd.DataFrame(columns=["end"])
    sequencesplit = []
    M = table.shape[0]

    # W_l=200
    E_l = 0
    for i in range(0, M):
        s_l = table.SEQUENCE[i].__len__()
        table_last = table_last.append(
            {"end": table.SEQUENCE[i][s_l - W_l : s_l - E_l]}, ignore_index=True
        )
    return table_last


def truncate_sequence_start(table, W_l=200):
    import pandas as pd

    table_last = pd.DataFrame(columns=["end"])
    sequencesplit = []
    M = table.shape[0]

    # W_l=200
    E_l = 0
    for i in range(0, M):
        s_l = table.SEQUENCE[i].__len__()
        table_last = table_last.append(
            {"end": table.SEQUENCE[i][E_l : E_l + W_l]}, ignore_index=True
        )
    return table_last


def truncate_sequence_s(table, W_l=200, E_l=0):
    import pandas as pd

    table_last = pd.DataFrame(columns=["end"])
    sequencesplit = []
    M = table.shape[0]

    # W_l=200

    for i in range(0, M):
        s_l = table.SEQUENCE[i].__len__()
        table_last = table_last.append(
            {"end": table.SEQUENCE[i][s_l - W_l - E_l : s_l - E_l]}, ignore_index=True
        )
    return table_last


def truncate_sequence_r(table, W_l=200, E_l=0):
    import pandas as pd

    table_last = pd.DataFrame(columns=["end"])
    sequencesplit = []
    M = table.shape[0]

    # W_l=200

    for i in range(0, M):
        s_l = table.SEQUENCE_INPUT[i].__len__()
        table_last = table_last.append(
            {"end": table.SEQUENCE_INPUT[i][s_l - W_l - E_l : s_l - E_l]},
            ignore_index=True,
        )
    return table_last


def truncate_sequence_rs(table, W_l=200):
    import pandas as pd

    table_last = pd.DataFrame(columns=["end"])
    sequencesplit = []
    M = table.shape[0]

    # W_l=200

    for i in range(0, M):
        s_l = table.values[i].__len__()
        table_last = table_last.append(
            {"end": table.values[i][:W_l]}, ignore_index=True
        )
    return table_last


def truncate_sequence_rf(table, W_l=200, E_l=0):
    import pandas as pd

    table_last = pd.DataFrame(columns=["end"])
    sequencesplit = []
    M = table.shape[0]

    # W_l=200

    for i in range(0, M):
        s_l = table.values[i].__len__()
        table_last = table_last.append(
            {"end": table.values[i][s_l - W_l - E_l : s_l - E_l]}, ignore_index=True
        )
    return table_last


def compute_dist2nearest(matches_fast):
    dist2nearestcosine = []
    for i in range(0, matches_fast.shape[0]):
        if sort_coo(matches_fast[i, :]).__len__() > 1:
            second_val = sort_coo(matches_fast[i, :])[1][1]
        else:
            second_val = 0
        dist2nearestcosine = np.append(dist2nearestcosine, 1 - second_val)

    return dist2nearestcosine


def cluster_from_matches(matches_fast, thresh_cosine):
    from scipy.sparse import find

    M = matches_fast.shape[0]
    M_L = 2000000
    clusters_cosine_full = np.array(list(range(M_L + 1, M + M_L + 1)))
    t = 0
    for i in range(0, M):
        matches_clust_ind = find(matches_fast[i, :] > 1 - thresh_cosine)
        if matches_clust_ind[1].shape[0] > 1:

            temp = np.where(clusters_cosine_full[matches_clust_ind[1]] < M_L)[0]
            if temp.shape[0] == 0:
                clusters_cosine_full[matches_clust_ind[1]] = t
                t = t + 1
            else:
                clusters_cosine_full[matches_clust_ind[1]] = clusters_cosine_full[
                    matches_clust_ind[1][temp[0]]
                ]
    return clusters_cosine_full


def assign_clones(matches_fast, thresh_cosine):
    from scipy.sparse import find

    M = matches_fast.shape[0]
    M_L = M + 1
    clusters_cosine_full = np.array(list(range(M_L + 1, M + M_L + 1)))
    t = 0
    for i in range(0, M):
        matches_clust_ind = find(matches_fast[i, :] > 1 - thresh_cosine)
        if matches_clust_ind[1].shape[0] > 1:

            temp = np.where(clusters_cosine_full[matches_clust_ind[1]] < M_L)[0]
            if temp.shape[0] == 0:
                clusters_cosine_full[matches_clust_ind[1]] = t
                t = t + 1
            else:
                clusters_cosine_full[matches_clust_ind[1]] = clusters_cosine_full[
                    matches_clust_ind[1][temp[0]]
                ]
    clusters_cosine_full[clusters_cosine_full > M_L] = (
        clusters_cosine_full[clusters_cosine_full > M_L]
        - M_L
        + np.max(clusters_cosine_full[clusters_cosine_full < M_L])
    )
    return clusters_cosine_full


def cluster_from_matches_vec(matches_fast, thresh_cosine_vec):
    from scipy.sparse import find

    M = matches_fast.shape[0]
    M_L = 2000000
    clusters_cosine_full = np.array(list(range(M_L + 1, M + M_L + 1)))
    t = 0
    for i in range(0, M):
        matches_clust_ind = find(matches_fast[i, :] > 1 - thresh_cosine_vec[i])
        if matches_clust_ind[1].shape[0] > 1:

            temp = np.where(clusters_cosine_full[matches_clust_ind[1]] < M_L)[0]
            if temp.shape[0] == 0:
                clusters_cosine_full[matches_clust_ind[1]] = t
                t = t + 1
            else:
                clusters_cosine_full[matches_clust_ind[1]] = clusters_cosine_full[
                    matches_clust_ind[1][temp[0]]
                ]
    return clusters_cosine_full


def clone_inference(table_clone_list, clusters_cosine_full):
    labels_list = np.zeros(table_clone_list.shape[0])
    for i in range(0, max(table_clone_list)):
        temp_in = np.where(table_clone_list == i)
        if temp_in[0].shape[0] > 1:
            labels_list[temp_in] = 1
    cloned_list = np.zeros(table_clone_list.shape[0])
    for i in range(0, max(clusters_cosine_full)):
        temp_in = np.where(clusters_cosine_full == i)
        if temp_in[0].shape[0] > 1:
            cloned_list[temp_in] = 1
    return labels_list, cloned_list


def sensitivity_comp(table_clone_list, clusters_cosine_full):
    sum_pairs = 0
    sum_clustered_pairs = 0
    for i in range(0, max(table_clone_list)):
        temp_in = np.where(table_clone_list == i)
        if temp_in[0].shape[0] > 1:
            combinations = list(itertools.combinations(temp_in[0], 2))
            sum_pairs = sum_pairs + combinations.__len__()
            per_clone_score = 0
            for j in range(0, combinations.__len__()):
                per_clone_score = per_clone_score + (
                    clusters_cosine_full[combinations[j][0]]
                    == clusters_cosine_full[combinations[j][1]]
                )
            sum_clustered_pairs = sum_clustered_pairs + per_clone_score
    sensitivity = sum_clustered_pairs / sum_pairs
    return sensitivity


def PPV_comp(table_clone_list, clusters_cosine_full):
    M_L = 2000000
    sum_pairs = 0
    sum_clustered_pairs = 0
    for i in range(0, max(clusters_cosine_full[np.where(clusters_cosine_full < M_L)])):
        temp_in = np.where(clusters_cosine_full == i)
        if temp_in[0].shape[0] > 1:
            combinations = list(itertools.combinations(temp_in[0], 2))
            sum_pairs = sum_pairs + combinations.__len__()
            per_clone_score = 0
            for j in range(0, combinations.__len__()):
                per_clone_score = per_clone_score + (
                    table_clone_list[combinations[j][0]]
                    == table_clone_list[combinations[j][1]]
                )
            sum_clustered_pairs = sum_clustered_pairs + per_clone_score
    PPV = sum_clustered_pairs / sum_pairs
    return PPV


def truncate_sequence_start_r(table, W_l=200):
    import pandas as pd

    table_last = pd.DataFrame(columns=["end"])
    sequencesplit = []
    M = table.shape[0]

    # W_l=200
    E_l = 0
    for i in range(0, M):
        s_l = table.SEQUENCE_INPUT[i].__len__()
        table_last = table_last.append(
            {"end": table.SEQUENCE_INPUT[i][E_l : E_l + W_l]}, ignore_index=True
        )
    return table_last


def cluster_from_matches_f(matches_fast, thresh_cosine):
    # from scipy.sparse import find
    M = matches_fast.shape[0]
    M_L = 2000000
    clusters_cosine_full = np.array(list(range(M_L + 1, M + M_L + 1)))
    t = 0
    for i in range(0, M):
        matches_clust_ind = np.where(matches_fast[i, :] < thresh_cosine)
        if matches_clust_ind[0].shape[0] > 1:

            temp = np.where(clusters_cosine_full[matches_clust_ind[0]] < M_L)[0]
            if temp.shape[0] == 0:
                clusters_cosine_full[matches_clust_ind[0]] = t
                t = t + 1
            else:
                clusters_cosine_full[matches_clust_ind[0]] = clusters_cosine_full[
                    matches_clust_ind[0][temp[0]]
                ]
    return clusters_cosine_full


def spec_vs_thresh_eval(
    dist2nearestcosine, dist2nearestcosine_neg_a, dist2nearestcosine_neg_b, labels_list
):
    spec_arr_var_a = []
    spec_arr_var_b = []
    for per in np.arange(0.1, 10, 0.1):
        thresh_cosine_b = np.percentile(dist2nearestcosine_neg_b, per)
        thresh_cosine_b = thresh_cosine_b * (
            1 - 0.09 * np.log(table_last.shape[0] / 5000)
        )
        specificity_b = np.sum(
            dist2nearestcosine[labels_list == 0] > thresh_cosine_b
        ) / sum(labels_list == 0)

        thresh_cosine_a = np.percentile(dist2nearestcosine_neg_a, per)
        thresh_cosine_a = thresh_cosine_a * (
            1 - 0.09 * np.log(table_last.shape[0] / 5000)
        )
        specificity_a = np.sum(
            dist2nearestcosine[labels_list == 0] > thresh_cosine_a
        ) / sum(labels_list == 0)
        spec_arr_var_b.append(specificity_b)
        spec_arr_var_a.append(specificity_a)
    return spec_arr_var_a, spec_arr_var_b


def compute_dist2nearest2(matches_fast):
    dist2nearestcosine = []
    for i in range(0, matches_fast.shape[0]):
        if sort_coo(matches_fast[i, :]).__len__() > 2:
            second_val = sort_coo(matches_fast[i, :])[2][1]
        else:
            second_val = 0
        dist2nearestcosine = np.append(dist2nearestcosine, 1 - second_val)

    return dist2nearestcosine


def get_lev_dist2nearest(table_full, clone_list, l):
    clone_indices = np.where(table_full.CLONE == clone_list[l])[0]

    dist2nearest_clone_min = 1000
    dist2nearest_clone_min_jun = 1000
    sum_non_unique_v = 0
    sum_non_unique_jun = 0
    gene_groups = np.unique(table_full.VJ_GROUP, return_counts=True)
    if gene_groups[1][0] > gene_groups[1][1]:
        min_group = np.where(table_full.VJ_GROUP[clone_indices] == gene_groups[0][1])[0]
        maj_group = np.where(table_full.VJ_GROUP[clone_indices] == gene_groups[0][0])[0]
    if gene_groups[1][0] < gene_groups[1][1]:
        min_group = np.where(table_full.VJ_GROUP[clone_indices] == gene_groups[0][0])[0]
        maj_group = np.where(table_full.VJ_GROUP[clone_indices] == gene_groups[0][1])[0]

    for i in range(0, min_group.shape[0]):
        for j in range(0, maj_group.shape[0]):
            d_ij = sift4(
                table_full.SEQUENCE_INPUT[clone_indices[min_group[i]]],
                table_full.SEQUENCE_INPUT[clone_indices[maj_group[j]]],
                max_offset=30,
            )
            d_ij = (
                2
                * (
                    d_ij
                    - np.abs(
                        table_full.SEQUENCE_INPUT[clone_indices[min_group[i]]].__len__()
                        - table_full.SEQUENCE_INPUT[
                            clone_indices[maj_group[j]]
                        ].__len__()
                    )
                )
                / (
                    table_full.SEQUENCE_INPUT[clone_indices[min_group[i]]].__len__()
                    + table_full.SEQUENCE_INPUT[clone_indices[maj_group[j]]].__len__()
                )
            )
            if d_ij < dist2nearest_clone_min:
                dist2nearest_clone_min = d_ij

            d_jun_ij = sift4(
                table_full.JUNCTION[clone_indices[min_group[i]]],
                table_full.JUNCTION[clone_indices[maj_group[j]]],
                max_offset=30,
            )
            # d_jun_ij=2*d_jun_ij/(d_jun_ij+table_full.JUNCTION[clone_indices[i]].__len__()+table_full.JUNCTION[clone_indices[j]].__len__() )
            d_jun_ij = (
                2
                * (
                    d_jun_ij
                    - np.abs(
                        table_full.JUNCTION[clone_indices[min_group[i]]].__len__()
                        - table_full.JUNCTION[clone_indices[maj_group[j]]].__len__()
                    )
                )
                / (
                    table_full.JUNCTION[clone_indices[min_group[i]]].__len__()
                    + table_full.JUNCTION[clone_indices[maj_group[j]]].__len__()
                )
            )
            if d_jun_ij < dist2nearest_clone_min_jun:
                dist2nearest_clone_min_jun = d_jun_ij

            if (
                table_full.JUNCTION[clone_indices[min_group[i]]].__len__()
                != table_full.JUNCTION[clone_indices[maj_group[j]]].__len__()
            ):
                sum_non_unique_jun = sum_non_unique_jun + 1

    return dist2nearest_clone_min, dist2nearest_clone_min_jun, sum_non_unique_jun
