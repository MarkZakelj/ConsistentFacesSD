import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

from utils import list_directories_in_directory
from utils.metrics import construct_dataframe, construct_dataframe_faces
from utils.paths import OUTPUT_DIR

# def create_bar_chart(df: pd.DataFrame, x: str, y: str, title: str):
#     fig = go.Figure()
#     fig.add_trace(go.Bar(
#         x=df[x],
#         y=df[y],
#         error_y=dict(
#             type='data',
#             array=standard_errors,
#             visible=True
#         ),
#         name='Data'
#     ))
#     return fig


def get_dataframe_faces(subset_name: str, identity_only: bool = True):
    df_raw = construct_dataframe_faces(subset_name)
    if identity_only:
        df_raw = df_raw.dropna(subset=["identity"])
    return df_raw


def get_dataframe(subset_name: str):
    df_raw = construct_dataframe(subset_name)
    return df_raw


def main():
    st.title("Metrics Visualizer")
    t1, t2, t3 = st.tabs(["face similarity", "clip score", "bbox sizes"])
    experiments = sorted(list_directories_in_directory(OUTPUT_DIR))
    experiments.remove("identities")

    with t1:
        st.write("Face similarity")
        selected_experiments_t1 = st.multiselect(
            "Select experiments", experiments, key="face_similarity"
        )
        if selected_experiments_t1:
            st.write(
                f"Selected experiments for face similarity: {', '.join(selected_experiments_t1)}"
            )
            dfs = pd.concat(
                [
                    get_dataframe_faces(subset_name)
                    for subset_name in selected_experiments_t1
                ],
                axis=0,
            )
            dfs = dfs.groupby("subset")["similarity_cosine"].agg(
                [
                    ("mean", "mean"),
                    ("std", "std"),
                    ("stderr", lambda x: x.std() / np.sqrt(x.count())),
                ]
            )
            st.dataframe(dfs)
            fig = px.bar(dfs, y="mean", error_y="stderr")
            st.plotly_chart(fig)
            # Add your logic here to process and display face similarity data for selected experiments

    with t2:
        st.write("Clip score")
        selected_experiments_t2 = st.multiselect(
            "Select experiments", experiments, key="clip_score"
        )
        if selected_experiments_t2:
            st.write(
                f"Selected experiments for clip score: {', '.join(selected_experiments_t2)}"
            )
            dfs = pd.concat(
                [get_dataframe(subset_name) for subset_name in selected_experiments_t2],
                axis=0,
            )
            dfs = dfs.groupby("subset")["clip_score"].agg(
                [
                    ("mean", "mean"),
                    ("std", "std"),
                    ("stderr", lambda x: x.std() / np.sqrt(x.count())),
                ]
            )
            st.dataframe(dfs)
            fig = px.bar(dfs, y="mean", error_y="stderr")
            st.plotly_chart(fig)
            # Add your logic here to process and display clip score data for selected experiments

    with t3:
        st.write("Bbox sizes")
        selected_experiments_t3 = st.multiselect(
            "Select experiments", experiments, key="bbox_sizes"
        )
        if selected_experiments_t3:
            st.write(
                f"Selected experiments for bbox sizes: {', '.join(selected_experiments_t3)}"
            )
            # Add your logic here to process and display bbox sizes data for selected experiments


if __name__ == "__main__":
    main()
