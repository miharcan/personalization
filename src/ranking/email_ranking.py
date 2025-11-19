import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from xgboost import XGBRanker

rng = np.random.default_rng(seed=42)

n_queries = 2000        # number of inbox views / users
n_per_query = 10        # candidates per query
n_samples = n_queries * n_per_query

query_ids = np.repeat(np.arange(n_queries), n_per_query)
email_ids = np.arange(n_samples)

# Features
from_vip = rng.binomial(1, 0.2, size=n_samples)               # 20% from VIP contacts
is_newsletter = rng.binomial(1, 0.3, size=n_samples)          # 30% newsletters
has_attachment = rng.binomial(1, 0.25, size=n_samples)        # 25% with attachment
recency_hours = rng.exponential(scale=24, size=n_samples)     # time since received
sender_reply_rate = rng.uniform(0, 1, size=n_samples)         # 0–1

df_rank = pd.DataFrame({
    "query_id": query_ids,
    "email_id": email_ids,
    "from_vip": from_vip,
    "is_newsletter": is_newsletter,
    "has_attachment": has_attachment,
    "recency_hours": recency_hours,
    "sender_reply_rate": sender_reply_rate,
})

# Higher score = more relevant
score = (
    2.0 * df_rank["from_vip"]
    - 1.0 * df_rank["is_newsletter"]
    + 0.5 * df_rank["has_attachment"]
    - 0.05 * df_rank["recency_hours"]
    + 1.5 * df_rank["sender_reply_rate"]
    + rng.normal(0, 0.5, size=n_samples)   # noise
)

df_rank["score_true"] = score

# Turn score into discrete relevance 0/1/2
# e.g. top 20% -> 2, next 30% -> 1, rest -> 0
q80 = df_rank["score_true"].quantile(0.8)
q50 = df_rank["score_true"].quantile(0.5)

def to_relevance(s):
    if s >= q80:
        return 2
    elif s >= q50:
        return 1
    else:
        return 0

df_rank["label"] = df_rank["score_true"].apply(to_relevance)
# print(df_rank.head())
# exit()

unique_queries = df_rank["query_id"].unique()
q_train, q_test = train_test_split(unique_queries, test_size=0.2, random_state=42)
# print(unique_queries.tolist())
# print(len(q_train.tolist()))
# exit()


train_mask = df_rank['query_id'].isin(q_train)
test_mask = df_rank['query_id'].isin(q_test)
# print(test_mask.value_counts())
# exit()


train_df = df_rank[train_mask].reset_index(drop=True)
test_df  = df_rank[test_mask].reset_index(drop=True)
# print(train_df)
# exit()


feature_cols = ["from_vip", "is_newsletter", "has_attachment",
                "recency_hours", "sender_reply_rate"]

X_train = train_df[feature_cols].values
y_train = train_df["label"].values
# print(y_train)
# exit()

X_test  = test_df[feature_cols].values
y_test  = test_df["label"].values

# group info: how many items per query, in order
group_train = train_df.groupby("query_id").size().to_list()
group_test  = test_df.groupby("query_id").size().to_list()


ranker = XGBRanker(
    objective="rank:pairwise",   # RankNet-style
    tree_method="hist",
    learning_rate=0.1,
    n_estimators=300,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

ranker.fit(
    X_train, y_train,
    group=group_train,
    eval_set=[(X_test, y_test)],
    eval_group=[group_test],
    verbose=False
)

def dcg_at_k(rels, k):
    rels = np.asarray(rels)[:k]
    if rels.size == 0:
        return 0.0
    discounts = 1.0 / np.log2(np.arange(2, rels.size + 2))
    return np.sum((2**rels - 1) * discounts)

def ndcg_at_k(y_true, y_scores, query_ids, k=5):
    df = pd.DataFrame({
        "q": query_ids,
        "y": y_true,
        "s": y_scores
    })
    ndcgs = []
    for q, group in df.groupby("q"):
        g_sorted = group.sort_values("s", ascending=False)
        ideal_sorted = group.sort_values("y", ascending=False)

        dcg = dcg_at_k(g_sorted["y"].values, k)
        idcg = dcg_at_k(ideal_sorted["y"].values, k)
        if idcg > 0:
            ndcgs.append(dcg / idcg)
    return np.mean(ndcgs)

y_scores = ranker.predict(X_test)
# print(X_test)
# print(y_scores)
# exit()

ndcg5 = ndcg_at_k(
    y_true=test_df["label"].values,
    y_scores=y_scores,
    query_ids=test_df["query_id"].values,
    k=5
)

print("NDCG@5:", ndcg5)

def precision_at_k_per_query(y_true, y_scores, query_ids, k=5):
    df = pd.DataFrame({"q": query_ids, "y": y_true, "s": y_scores})
    precisions = []
    for q, group in df.groupby("q"):
        g_sorted = group.sort_values("s", ascending=False)
        topk = g_sorted["y"].values[:k]
        precisions.append((topk > 0).mean())
    return np.mean(precisions)

print("Precision@5:", precision_at_k_per_query(
    test_df["label"].values, y_scores, test_df["query_id"].values, k=5
))

some_q = test_df["query_id"].iloc[0]
g = test_df[test_df["query_id"] == some_q].copy()
g["pred"] = ranker.predict(g[feature_cols].values)
print(g.sort_values("pred", ascending=False)[
    ["query_id", "email_id", "from_vip", "is_newsletter",
     "has_attachment", "recency_hours", "sender_reply_rate",
     "label", "pred"]
])
