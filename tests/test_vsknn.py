from vs_knn.vs_knn import CupyVsKnnModel
import cudf
import cupy as cp
import asyncio
from vs_knn.col_names import SESSION_ID, ITEM_ID


def shuffle_df(df: cudf.DataFrame):
    """Test results should be unaffected by row ordering """
    return df.sample(frac=1).reset_index().drop(columns=['index'])


def test_cupymodel(tiny_vsknn_df, tiny_session):
    # TODO test what happens if none of the query values are found in mapping
    # TODO: test with top_k > n_sessions
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
    for idx in range(int(predicted_items.shape[0])):
        predicted_score_py = [float(s) for s in predicted_score[idx]]
        assert all([pred == expected for pred, expected in zip(predicted_items[idx], [1, 2, 3, 4, 5, 6])])
        assert all([abs(score - expected) < 0.01 for score, expected
                    in zip(predicted_score_py, [2.0, 1.666, 2.0, 3.666, 3.666, 1.666])])


def test_unique_per_row():
    """
    unique per col for
    [[4, 1, 3, 0, 0,
      3, 3, 0, 0, 0,
      1, 1, 1, 0, 0,
      1, 2, 1, 0, 0,
    ]]
    :return:
    """
    test2d = cp.array([[4, 1, 3, 0, 0],
                        [3, 3, 0, 0, 0],
                        [1, 1, 1, 0, 0],
                        [1, 2, 1, 0, 0],
                        ])
    model = CupyVsKnnModel()
    res = model._unique_per_row(test2d)
    assert res.shape[1] == test2d.shape[1] - 2


def test_keep_topk():
    """
    sessions:
    [[1, 2, 3, 4],
    [1, 2, 3, 4]]
    similarities:
    [[1., 2., 3., 4.],
    [4., 3., 2., 1.]]
    top_k = 2
    expected results:
     sessions:   [[1, 2],
                 [4, 3]]
     similarities:  [[4., 3.],
                    [4., 3.]]
    """
    sessions = cp.vstack([cp.arange(1, 5), cp.arange(1, 5)])
    session_similarities = cp.vstack([
        cp.arange(1, 5, dtype=cp.float32),
        cp.flip(cp.arange(1, 5, dtype=cp.float32))])
    model = CupyVsKnnModel(top_k=2)
    sess, sims = model.keep_topk_sessions(sessions, session_similarities)
    for i in range(len(sess[0])):
        assert sess[0, i] == int(sims[0, i])
        assert sess[1, i] == 5 - int(sims[1, i])


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
