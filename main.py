import glob
from itertools import cycle
from os import path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objs as go
import streamlit as st
from gensim.models import KeyedVectors, Word2Vec
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KDTree
from wordcloud import WordCloud


@st.cache
def clustering_on_wordvecs(word_vectors, num_clusters):
    kmeans_clustering = KMeans(n_clusters=num_clusters, init="k-means++")
    kmeans_clustering.fit_predict(word_vectors)

    return kmeans_clustering.cluster_centers_


@st.cache
def get_top_words(index2word, k, centers, wordvecs):
    tree = KDTree(wordvecs)
    closest_points = [tree.query(np.reshape(x, (1, -1)), k=k) for x in centers]

    closest_words_idxs = [x[1] for x in closest_points]
    closest_words = {}
    for i in range(0, len(closest_words_idxs)):
        closest_words["Cluster #" + str(i)] = [
            str(index2word[j]) for j in closest_words_idxs[i][0]
        ]
    df = pd.DataFrame(closest_words)
    df.index = df.index + 1

    return df


@st.cache
def generate_cloud(cluster_num, cmap, top_words, background_color="black"):
    wc = WordCloud(
        width=800,
        height=400,
        background_color=background_color,
        max_words=2000,
        max_font_size=80,
        colormap=cmap,
        stopwords=[],
        include_numbers=True,
    )
    return wc.generate(
        " ".join([word for word in top_words["Cluster #" + str(cluster_num)]])
    )


@st.cache
def get_cloud(cluster_num, cmap, top_words, background_color="black"):
    wordcloud = generate_cloud(cluster_num, cmap, top_words, background_color)
    fig = px.imshow(wordcloud)
    fig.update_layout(width=800, height=400, margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    return fig


def display_cloud(cluster_num, cmap, top_words, background_color="black"):
    st.write("Cluster #" + str(cluster_num + 1))
    st.plotly_chart(get_cloud(cluster_num, cmap, top_words, background_color))


@st.cache
def generate_cloud_from_frequency(words, cmap, background_color="black"):
    wc = WordCloud(
        width=800,
        height=400,
        background_color=background_color,
        max_words=2000,
        max_font_size=80,
        colormap=cmap,
        relative_scaling=0.5,
    )
    return wc.generate_from_frequencies(words)


def display_cloud_from_frequency(words, cmap, background_color="black"):
    wordcloud = generate_cloud_from_frequency(words, cmap, background_color)
    fig = px.imshow(wordcloud)
    fig.update_layout(width=800, height=400, margin=dict(l=0, r=0, b=0, t=0))
    fig.update_layout(coloraxis_showscale=False)
    fig.update_xaxes(showticklabels=False)
    fig.update_yaxes(showticklabels=False)
    st.plotly_chart(fig)


@st.cache
def append_list(sim_words, words):
    list_of_words = []

    for i in range(len(sim_words)):
        sim_words_list = list(sim_words[i])
        sim_words_list.append(words)
        sim_words_tuple = tuple(sim_words_list)
        list_of_words.append(sim_words_tuple)

    return list_of_words


@st.cache
def calc_3d_pca(word_vectors):
    return PCA(random_state=0).fit_transform(word_vectors)[:, :3]


@st.cache
def calc_3d_tsne(word_vectors, perplexity, learning_rate, n_iter):
    return TSNE(
        n_components=3,
        random_state=0,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
    ).fit_transform(word_vectors)[:, :3]


@st.cache
def generate_scatterplot_3D(
    word_vectors,
    user_input=None,
    words=None,
    annotation="On",
    dim_red="PCA",
    perplexity=0,
    learning_rate=0,
    iteration=0,
    topn=0,
):
    if dim_red == "PCA":
        three_dim = calc_3d_pca(word_vectors)
    else:
        three_dim = calc_3d_tsne(word_vectors, perplexity, learning_rate, iteration)

    color = "blue"
    quiver = go.Cone(
        x=[0, 0, 0],
        y=[0, 0, 0],
        z=[0, 0, 0],
        u=[1.5, 0, 0],
        v=[0, 1.5, 0],
        w=[0, 0, 1.5],
        anchor="tail",
        colorscale=[[0, color], [1, color]],
        showscale=False,
    )

    data = [quiver]

    count = 0
    for i in range(len(user_input)):

        trace = go.Scatter3d(
            x=three_dim[count : count + topn, 0],
            y=three_dim[count : count + topn, 1],
            z=three_dim[count : count + topn, 2],
            text=words[count : count + topn] if annotation == "On" else "",
            name=user_input[i],
            textposition="top center",
            textfont_size=30,
            mode="markers+text",
            marker={"size": 10, "opacity": 0.8, "color": 2},
        )

        data.append(trace)
        count = count + topn

    trace_input = go.Scatter3d(
        x=three_dim[count:, 0],
        y=three_dim[count:, 1],
        z=three_dim[count:, 2],
        text=words[count:],
        name="input words",
        textposition="top center",
        textfont_size=30,
        mode="markers+text",
        marker={"size": 10, "opacity": 1, "color": "black"},
    )

    data.append(trace_input)

    # Configure the layout.
    layout = go.Layout(
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        showlegend=True,
        legend=dict(
            x=1, y=0.5, font=dict(family="Courier New", size=24, color="black")
        ),
        font=dict(family=" Courier New ", size=15),
        autosize=False,
        width=800,
        height=600,
    )

    return go.Figure(data=data, layout=layout)


def display_scatterplot_3D(
    model,
    user_input=None,
    words=None,
    annotation="On",
    dim_red="PCA",
    perplexity=0,
    learning_rate=0,
    iteration=0,
    topn=0,
):
    plot_figure = generate_scatterplot_3D(
        np.array([model[w] for w in words]),
        user_input,
        words,
        annotation,
        dim_red,
        perplexity,
        learning_rate,
        iteration,
        topn,
    )

    st.plotly_chart(plot_figure)


@st.cache
def calc_2d_pca(word_vectors):
    return PCA(random_state=0).fit_transform(word_vectors)[:, :2]


@st.cache
def calc_2d_tsne(word_vectors, perplexity, learning_rate, n_iter):
    return TSNE(
        random_state=0,
        perplexity=perplexity,
        learning_rate=learning_rate,
        n_iter=n_iter,
    ).fit_transform(word_vectors)[:, :2]


@st.cache
def generate_scatterplot_2D(
    word_vectors,
    user_input=None,
    words=None,
    annotation="On",
    dim_red="PCA",
    perplexity=0,
    learning_rate=0,
    iteration=0,
    topn=0,
):
    if dim_red == "PCA":
        two_dim = calc_2d_pca(word_vectors)
    else:
        two_dim = calc_2d_tsne(word_vectors, perplexity, learning_rate, iteration)

    data = []
    count = 0
    for i in range(len(user_input)):

        trace = go.Scatter(
            x=two_dim[count : count + topn, 0],
            y=two_dim[count : count + topn, 1],
            text=words[count : count + topn] if annotation == "On" else "",
            name=user_input[i],
            textposition="top center",
            textfont_size=20,
            mode="markers+text",
            marker={"size": 15, "opacity": 0.8, "color": 2},
        )

        data.append(trace)
        count = count + topn

    trace_input = go.Scatter(
        x=two_dim[count:, 0],
        y=two_dim[count:, 1],
        text=words[count:],
        name="input words",
        textposition="top center",
        textfont_size=20,
        mode="markers+text",
        marker={"size": 25, "opacity": 1, "color": "black"},
    )

    data.append(trace_input)

    # Configure the layout.
    layout = go.Layout(
        margin={"l": 0, "r": 0, "b": 0, "t": 0},
        showlegend=True,
        hoverlabel=dict(bgcolor="white", font_size=20, font_family="Courier New"),
        legend=dict(
            x=1, y=0.5, font=dict(family="Courier New", size=24, color="black")
        ),
        font=dict(family=" Courier New ", size=15),
        autosize=False,
        width=800,
        height=600,
    )

    return go.Figure(data=data, layout=layout)


def display_scatterplot_2D(
    model,
    user_input=None,
    words=None,
    annotation="On",
    dim_red="PCA",
    perplexity=0,
    learning_rate=0,
    iteration=0,
    topn=0,
):
    plot_figure = generate_scatterplot_2D(
        np.array([model[w] for w in words[: len(words) - len(user_input)]]),
        user_input,
        words,
        annotation,
        dim_red,
        perplexity,
        learning_rate,
        iteration,
        topn,
    )

    st.plotly_chart(plot_figure)


@st.cache()
def generate_horizontal_bar_plot(word, similarity):
    similarity = [round(elem, 2) for elem in similarity]

    data = go.Bar(
        x=similarity,
        y=word,
        orientation="h",
        text=similarity,
        marker_color=4,
        textposition="auto",
    )

    layout = go.Layout(
        font=dict(size=20),
        xaxis=dict(showticklabels=False, automargin=True),
        yaxis=dict(showticklabels=True, automargin=True, autorange="reversed"),
        margin=dict(t=20, b=20, r=10),
    )

    return go.Figure(data=data, layout=layout)


def horizontal_bar(word, similarity):
    fig = generate_horizontal_bar_plot(word, similarity)
    st.plotly_chart(fig)


@st.cache(allow_output_mutation=True)
def get_model(gensim_model, model_types, models_path, limit_vectors):
    if model_types[gensim_model] == "full Gensim model":
        return Word2Vec.load(models_path + gensim_model + ".model", mmap="r").wv
    elif model_types[gensim_model] == "word2vec binary model":
        if limit_vectors is None or limit_vectors == "Yes":
            return KeyedVectors.load_word2vec_format(
                models_path + gensim_model + ".bin", binary=True, limit=500000
            )
        else:
            return KeyedVectors.load_word2vec_format(
                models_path + gensim_model + ".bin", binary=True
            )
    elif model_types[gensim_model] == "Gensim kv model":
        return KeyedVectors.load(models_path + gensim_model + ".kv", mmap="r")


# Seems to be faster without cache, probably because of model passing model
def get_similarity_matrix_figure(model, similarity_matrix_input):
    x = similarity_matrix_input
    y = similarity_matrix_input

    z = []
    # TODO: Performance optimization necessary
    for xx in x:
        row = []
        for yy in x:
            row.append(model.similarity(xx, yy))
        z.append(row)

    fig = px.imshow(z, labels=dict(x="Word", y="Word", color="Similarity"), x=x, y=y)
    fig.update_xaxes(side="top")
    fig.update_layout(width=800, height=600, margin=dict(l=0, r=0, b=0, t=0))

    return fig


def main():
    models_path = "./models/"

    models = [
        model.rstrip(".model").lstrip(models_path)
        for model in glob.glob(models_path + "*.model")
    ]

    bin_models = [
        model.rstrip(".bin").lstrip(models_path)
        for model in glob.glob(models_path + "*.bin")
    ]

    kv_models = [
        model.rstrip(".kv").lstrip(models_path)
        for model in glob.glob(models_path + "*.kv")
    ]

    model_descriptions = {}
    model_types = {}
    for model in models:
        model_types[model] = "full Gensim model"
        description_file = models_path + model + ".txt"
        if path.isfile(description_file):
            with open(description_file, "r") as file:
                description = file.read()
                model_descriptions[model] = description.rstrip("\n")
        else:
            model_descriptions[model] = f"No model description found for **{model}**"

    for model in bin_models:
        model_types[model] = "word2vec binary model"
        description_file = models_path + model + ".txt"
        if path.isfile(description_file):
            with open(description_file, "r") as file:
                description = file.read()
                model_descriptions[model] = description.rstrip("\n")
        else:
            model_descriptions[model] = f"No model description found for **{model}**"

    for model in kv_models:
        model_types[model] = "Gensim kv model"
        description_file = models_path + model + ".txt"
        if path.isfile(description_file):
            with open(description_file, "r") as file:
                description = file.read()
                model_descriptions[model] = description.rstrip("\n")
        else:
            model_descriptions[model] = f"No model description found for **{model}**"

    models += bin_models + kv_models

    gensim_model = st.sidebar.selectbox("Word2Vec model to use:", models)

    if model_types[gensim_model] == "word2vec binary model":
        limit_vectors = st.sidebar.radio(
            "Limit the amount of vectors loaded. Warning: If turned off may need tons of memory!",
            ("Yes", "No"),
        )
    else:
        limit_vectors = "No"

    model = get_model(gensim_model, model_types, models_path, limit_vectors)

    most_similar_method = st.sidebar.selectbox(
        "Similarity method:", ("Cosine", "3CosMul")
    )

    user_input = st.sidebar.text_input(
        "Type the word that you want to investigate. Separate more than one word by semicolon (;). You can also group words like [woman, king, -man] where words with '-' in front count negatively.",
        "",
    )

    top_n = st.sidebar.slider(
        "Select the amount of words associated with the input words you want to visualize:",
        5,
        100,
        (5),
    )

    dimension = st.sidebar.radio("Dimension of the visualization:", ("2D", "3D"))

    dim_red = st.sidebar.selectbox("Dimension reduction method:", ("PCA", "t-SNE"))

    if dim_red == "t-SNE":
        perplexity = st.sidebar.slider(
            "t-SNE - Perplexity (It says (loosely) how to balance attention between local and global aspects of your data. Larger datasets usually require a larger perplexity):",
            5,
            50,
            (30),
        )

        learning_rate = st.sidebar.slider("t-SNE - Learning rate:", 10, 1000, (200))

        iteration = st.sidebar.slider(
            "t-SNE - Number of iteration:", 250, 100000, (1000)
        )

    else:
        perplexity = 0
        learning_rate = 0
        iteration = 0

    annotation = st.sidebar.radio(
        "Enable or disable the annotation on the visualization:", ("On", "Off")
    )

    word_cloud_background_color = st.sidebar.radio(
        "Word cloud background color:", ("black", "white")
    )

    show_top_5_most_similar_words = st.sidebar.radio(
        "Enable or disable showing the top 5 most similar words and similarity word clouds for the words you entered:",
        ("On", "Off"),
    )
    show_kmeans_cluster_word_clouds = st.sidebar.radio(
        "Enable or disable k-means cluster calculation:", ("On", "Off")
    )

    if show_kmeans_cluster_word_clouds == "On":
        n_kmeans_clusters = st.sidebar.slider(
            "k-means - Number of clusters:", 1, 20, (5)
        )
        n_words_per_kmeans_cluster = st.sidebar.slider(
            "k-means - Number of words per cluster to show:", 1, 20, (10)
        )

    show_similarity_matrix = st.sidebar.radio(
        "Enable or disable similarity matrix:", ("On", "Off"), index=1
    )

    if show_similarity_matrix == "On":
        similarity_matrix_input = st.sidebar.text_input(
            "Type the words that you want to have in the similarity matrix. Separate more than one word by comma (,).",
        )
        if similarity_matrix_input.strip() != "":
            similarity_matrix_input = [
                x.strip() for x in similarity_matrix_input.strip().split(",")
            ]
        else:
            similarity_matrix_input = []
    else:
        similarity_matrix_input = []

    st.title("Visualization of word embeddings")

    model_description = (
        "**"
        + gensim_model
        + "** ("
        + model_types[gensim_model]
        + "): "
        + model_descriptions[gensim_model]
    )

    if user_input == "":
        similar_word = None

        st.markdown(
            "Word embedding, in natural language processing, is a representation of the meaning of words. It can be obtained using a set of language modeling and feature learning techniques where words or phrases from the vocabulary are mapped to vectors of real numbers. For more details visit https://en.wikipedia.org/wiki/Word_embedding"
        )

        st.markdown(
            "You can use the sidebar to chose between different models, visualizations and options."
        )

        st.markdown(
            "You can type the words you want to investigate into the sidebar. Separate more than one word by semicolon (;). You can also group words like [woman, king, -man] where words with '-' in front count negatively."
        )

        st.markdown(
            "With the slider in the sidebar, you can pick the amount of words associated with the input word you want to visualize. The words to show is determined by either the cosine or 3CosMul similarity between the word vectors."
        )

        st.markdown(
            "There is also the option to generate a similarity matrix in the sidebar. When enabled you can enter a separate set of words and a heatmap visualizing the similarties between the words is shown."
        )

        st.header("Description of selected model")
        st.markdown(model_description)

    else:
        st.header("Description of selected model")
        st.markdown(model_description)

        user_input = [x.strip() for x in user_input.split(";")]
        input_groups = {}
        for input in user_input:
            if input[0] == "[" and input[len(input) - 1] == "]":
                input_splitted = input.replace("[", "").replace("]", "").split(",")
                all = [x.strip() for x in input_splitted]
                positive = []
                negative = []
                for one in all:
                    if one[0] == "+":
                        positive.append(one.strip("+"))
                    elif one[0] == "-":
                        negative.append(one.strip("-"))
                    else:
                        positive.append(one)
                input_groups[input] = {"positive": positive, "negative": negative}
            else:
                input_groups[input] = {"positive": [input], "negative": []}
        result_word = []

        # TODO: Kmeans cluster visualization in a plot?

        for words in user_input:

            if most_similar_method == "3CosMul":
                sim_words = model.most_similar_cosmul(
                    positive=input_groups[words]["positive"],
                    negative=input_groups[words]["negative"],
                    topn=top_n,
                )
            else:
                sim_words = model.most_similar(
                    positive=input_groups[words]["positive"],
                    negative=input_groups[words]["negative"],
                    topn=top_n,
                )
            sim_words = append_list(sim_words, words)

            result_word.extend(sim_words)

        similar_word = [word[0] for word in result_word]
        similarity = [word[1] for word in result_word]
        similar_word.extend(user_input)

        if show_kmeans_cluster_word_clouds == "On":
            Z = model.vectors

            centers = clustering_on_wordvecs(Z, n_kmeans_clusters)

            top_words = get_top_words(
                model.index2word, n_words_per_kmeans_cluster, centers, Z
            )

        if dimension == "2D":
            st.header("2D " + dim_red + " Visualization")
            if dim_red != "PCA":
                st.markdown(
                    "Try playing around with the t-SNE hyperparameters in the sidebar, they really matter!"
                )
            display_scatterplot_2D(
                model,
                user_input,
                similar_word,
                annotation,
                dim_red,
                perplexity,
                learning_rate,
                iteration,
                top_n,
            )
        else:
            st.header("3D " + dim_red + " Visualization")
            if dim_red != "PCA":
                st.markdown(
                    "Try playing around with the t-SNE hyperparameters in the sidebar, they really matter!"
                )
            display_scatterplot_3D(
                model,
                user_input,
                similar_word,
                annotation,
                dim_red,
                perplexity,
                learning_rate,
                iteration,
                top_n,
            )

        cmaps = cycle(
            [
                "flag",
                "prism",
                "ocean",
                "gist_earth",
                "terrain",
                "gist_stern",
                "gnuplot",
                "gnuplot2",
                "CMRmap",
                "cubehelix",
                "brg",
                "hsv",
                "gist_rainbow",
                "rainbow",
                "jet",
                "nipy_spectral",
                "gist_ncar",
            ]
        )

        if show_top_5_most_similar_words == "On":
            st.header("The Top 5 Most Similar Words for Each Input")
            count = 0
            for i in range(len(user_input)):

                st.markdown(
                    "The most similar words to *"
                    + str(user_input[i])
                    + "* and their similarity are:"
                )
                horizontal_bar(
                    similar_word[count : count + 5], similarity[count : count + 5]
                )

                count = count + top_n

            st.header("Word clouds of the most similar words for each input")
            count = 0
            for i in range(len(user_input)):
                st.write("Word cloud for *" + str(user_input[i]) + "*")
                words = []
                sum = 0
                for i in range(count, count + top_n):
                    words += [(similar_word[i], similarity[i])]
                    sum += similarity[i]
                df = pd.DataFrame(words, columns=["word", "similarity"])
                df["similarity"] = df["similarity"] * 100
                df["similarity"] = df["similarity"].astype("int")
                words = {}
                for index, r in df.iterrows():
                    words[str(r["word"])] = r["similarity"]
                display_cloud_from_frequency(
                    words, next(cmaps), word_cloud_background_color
                )

                count = count + top_n

        if show_kmeans_cluster_word_clouds == "On":
            st.header("k-means clusters")
            st.write(
                "This visualization shows the n closest words to the determined k-means cluster centers for each cluster. Use the sidebar to adjust the number of clusters and the number of words per cluster to show."
            )

            for i in range(n_kmeans_clusters):
                col = next(cmaps)
                display_cloud(i, col, top_words, word_cloud_background_color)

    if show_similarity_matrix == "On":
        st.header("Similarity matrix")
        st.write(
            "This visualization shows a heatmap representation of the similarities between the entered words."
        )
        if len(similarity_matrix_input) > 0:
            fig = get_similarity_matrix_figure(model, similarity_matrix_input)
            st.plotly_chart(fig)
        else:
            st.write("Use the sidebar to enter words for the similarity matrix.")


if __name__ == "__main__":
    main()
