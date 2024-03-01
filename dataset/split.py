import numpy as np
import pandas as pd
from argparse import ArgumentParser

def main(args):
    df = pd.read_csv(args.raw_data_path)
    
    if args.proportional:
        k_mbtis = {"ISFJ": 0.0908, "ISTJ": 0.0889, "INFP": 0.0807, "INFJ": 0.0768, 
                "ENFP": 0.0736, "ISFP": 0.0713, "ENFJ": 0.0661, "ESFJ": 0.0631,
                "ESTJ": 0.0611, "ISTP": 0.0537, "INTJ": 0.0597, "ESFP": 0.0521,
                "INTP": 0.0492, "ENTJ": 0.0487, "ENTP": 0.0361, "ESTP": 0.0281}
        real_fracs = pd.DataFrame(k_mbtis, index=["real_frac"]).transpose().reset_index().rename(columns={"index":"type"})
        
        counts = pd.DataFrame(df.groupby("type").count()).reset_index()
        all = counts.merge(real_fracs, on="type", how="left")
        maximal_count = int((all["posts"]/all["real_frac"]).min())
        all["sample_cnt"] = (all["real_frac"] * maximal_count).astype(int)
        
        resampled_df = pd.DataFrame(columns=["posts", "type"])
        for i in range(all.shape[0]):
            add_df = df[df["type"] == all["type"].iloc[i]].sample(all["sample_cnt"].iloc[i], replace=False)
            resampled_df = pd.concat([resampled_df, add_df], axis=0)
        print(f"Resampled df size: {resampled_df.shape[0]}")
        
        df = resampled_df

    if args.balance:
        k_mbtis = {"ISFJ": 0.0625, "ISTJ": 0.0625, "INFP": 0.0625, "INFJ": 0.0625, 
                "ENFP": 0.0625, "ISFP": 0.0625, "ENFJ": 0.0625, "ESFJ": 0.0625,
                "ESTJ": 0.0625, "ISTP": 0.0625, "INTJ": 0.0625, "ESFP": 0.0625,
                "INTP": 0.0625, "ENTJ": 0.0625, "ENTP": 0.0625, "ESTP": 0.0625}
        real_fracs = pd.DataFrame(k_mbtis, index=["real_frac"]).transpose().reset_index().rename(columns={"index":"type"})
        
        counts = pd.DataFrame(df.groupby("type").count()).reset_index()
        all = counts.merge(real_fracs, on="type", how="left")
        maximal_count = int((all["posts"]/all["real_frac"]).min())
        all["sample_cnt"] = (all["real_frac"] * maximal_count).astype(int)
        
        resampled_df = pd.DataFrame(columns=["posts", "type"])
        for i in range(all.shape[0]):
            add_df = df[df["type"] == all["type"].iloc[i]].sample(all["sample_cnt"].iloc[i], replace=False)
            resampled_df = pd.concat([resampled_df, add_df], axis=0)
        print(f"Resampled df size: {resampled_df.shape[0]}")
        
        df = resampled_df
    
    train_idx = np.random.choice(np.arange(df.shape[0]), size = int(df.shape[0] * args.train_ratio), replace=False)
    test_idx = np.array(list(set(list(np.arange(df.shape[0]))).difference(set(train_idx.tolist()))))
    
    train_df = df.iloc[train_idx]
    test_df = df.iloc[test_idx]

    if args.proportional:
        train_df.to_csv(f"{args.train_data_path}_prop.csv", index=False)
        test_df.to_csv(f"{args.eval_data_path}_prop.csv", index=False)        
    elif args.balance:
        train_df.to_csv(f"{args.train_data_path}_bal.csv", index=False)
        test_df.to_csv(f"{args.eval_data_path}_bal.csv", index=False)                
    else:
        train_df.to_csv(f"{args.train_data_path}.csv", index=False)
        test_df.to_csv(f"{args.eval_data_path}.csv", index=False)

    print(f"train_df of size {train_df.shape[0]} saved!")    
    print(f"test_df of size {test_df.shape[0]} saved!")

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--raw_data_path", type=str, default="dataset/MBTI 500.csv")
    parser.add_argument("--train_data_path", type=str, default="dataset/MBTI 500_train")
    parser.add_argument("--eval_data_path", type=str, default="dataset/MBTI 500_eval")
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--balance", type=bool, default=False)
    parser.add_argument("--proportional", type=bool, default=False)
    args = parser.parse_args() 
    main(args)