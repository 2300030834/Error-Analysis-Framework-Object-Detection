def calculate_error_impact(metric_name, metric, errors_df, targets_df, preds_df):

    baseline = metric(targets_df, preds_df)

    impact = {}

    error_types = ["CLS","LOC","BKG","MISS"]

    for err in error_types:

        fixed_preds = preds_df.copy()

        bad_preds = errors_df[errors_df["error_type"] == err]["pred_id"]

        fixed_preds = fixed_preds[~fixed_preds["pred_id"].isin(bad_preds)]

        new_score = metric(targets_df, fixed_preds)

        impact[err] = new_score - baseline

    impact[metric_name] = baseline

    return impact