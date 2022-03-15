from vs_knn.vs_knn import CupyVsKnnModel
import cudf
import asyncio
from vs_knn.col_names import SESSION_ID, ITEM_ID


def shuffle_df(df: cudf.DataFrame):
    """Test results should be unaffected by row ordering """
    return df.sample(frac=1).reset_index().drop(columns=['index'])


def test_cupymodel(tiny_vsknn_df, tiny_session):
    #TODO test what happens if none of the query values are found in mapping
    model = CupyVsKnnModel(top_k=2, max_sessions_per_items=20)
    model.train(tiny_vsknn_df)
    model_predict_test(model, tiny_session)


def test_cupymodel_batch(tiny_vsknn_df, tiny_batch):
    model = CupyVsKnnModel(top_k=2, max_sessions_per_items=20)
    model.train(tiny_vsknn_df)
    model_predict_test(model, tiny_batch)


def model_predict_test(model: CupyVsKnnModel, tiny_session):
    loop = asyncio.get_event_loop()
    coroutine = model.predict(tiny_session)
    predictions = loop.run_until_complete(coroutine)
    predicted_items, predicted_score = predictions['predicted_items'], predictions['scores']
    predicted_score_py = [float(s) for s in predicted_score]
    assert all([pred == expected for pred, expected in zip(predicted_items, [1, 2, 3, 4, 5, 6])])
    assert all([abs(score - expected) < 0.01 for score, expected
                in zip(predicted_score_py, [2.0, 1.666, 2.0, 3.666, 3.666, 1.666])])


def test_save_load(tiny_vsknn_df, tiny_session, tmpdir):
    model = CupyVsKnnModel(top_k=2, max_sessions_per_items=20)
    model.train(tiny_vsknn_df)

    model.save(tmpdir)
    model_new = CupyVsKnnModel()
    model_new.load(tmpdir)

    model_predict_test(model_new, tiny_session)


def test_cupymodel_str(tiny_vsknn_df, tiny_session, tmpdir):
    tiny_vsknn_df["session_col"] = 'sess_' + tiny_vsknn_df[SESSION_ID].astype(str)
    tiny_vsknn_df["item_col"] = 'item_' + tiny_vsknn_df[ITEM_ID].astype(str)

    model = CupyVsKnnModel(top_k=2, max_sessions_per_items=20, item_col="item_col", session_col="session_col")
    model.train(tiny_vsknn_df)

    model.save(tmpdir)
    model_new = CupyVsKnnModel()
    model_new.load(tmpdir)

    model_predict_test(model_new, tiny_session)
