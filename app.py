import streamlit as st
from project import pixel_wise_matching, window_based_matching, window_based_matching_cosine
from utils import compute_distance_l1, compute_distance_l2
from PIL import Image

### Title of the app ###

st.title("✨Project-Module2: Depth Information Estimation")

### Sidebar ###
## Parameters ##
st.sidebar.markdown("### Choosing parameters")
method = st.sidebar.selectbox(
    "Method", ["Pixel-wise", "Window-based", "Window-based with cosine similarity"])
disparty_range = st.sidebar.slider(
    "Disparity range", min_value=0, max_value=64, value=64)
kernel_size = st.sidebar.slider(
    "Kernel size", min_value=1, max_value=10, value=3)
scale = st.sidebar.slider("Scale", min_value=1, max_value=10, value=3)

## Upload images ##
st.sidebar.markdown("### Please upload two images")
left_img_uploader = st.sidebar.file_uploader("Left image", type=["png", "jpg"])
right_img_uploader = st.sidebar.file_uploader(
    "Right image", type=["png", "jpg"])

st.sidebar.markdown("### Or select sample images")
sample_img = st.sidebar.selectbox(
    "Sample image", ["none", "aloe.png", "tsukuba.png"])


btn_upload = st.sidebar.button("Load Image")


### Main ###

### Display images ##
if 'uploaded_1' not in st.session_state and 'uploaded_2' not in st.session_state:
    st.session_state.uploaded_1 = False
    st.session_state.uploaded_2 = False

col1, col2 = st.columns(2)
with col1:
    left_img_main = st.image("data\\placeholder.png", use_column_width=True)
with col2:
    right_img_main = st.image("data\\placeholder.png", use_column_width=True)

if btn_upload:
    if left_img_uploader is not None:
        st.session_state.uploaded_1 = True
    if right_img_uploader is not None:
        st.session_state.uploaded_2 = True
    if sample_img != "none" and (left_img_uploader or right_img_uploader) is None:
        st.session_state.uploaded_1 = True
        st.session_state.uploaded_2 = True


if st.session_state.uploaded_1:
    if sample_img != "none":
        if sample_img == "aloe.png":
            left_img_main.image(
                "data\\aloe\\Aloe_left_1.png", use_column_width=True)
        else:
            left_img_main.image("data\\tsukuba\\left.png",
                                use_column_width=True)
    else:
        left_img_main.image(left_img_uploader, use_column_width=True)

if st.session_state.uploaded_2:
    if sample_img != "none":
        if sample_img == "aloe.png":
            right_img_main.image(
                "data\\aloe\\Aloe_right_1.png", use_column_width=True)
        else:
            right_img_main.image(
                "data\\tsukuba\\right.png", use_column_width=True)
    else:
        right_img_main.image(right_img_uploader, use_column_width=True)

### Display results ##

btn_results = st.button("Show results")
if btn_results and st.session_state.uploaded_1 and st.session_state.uploaded_2:
    print("Show results")
    if (left_img_uploader or right_img_uploader) is None:
        if sample_img == "aloe.png":
            left_img = Image.open("data\\aloe\\Aloe_left_1.png")
            right_img = Image.open("data\\aloe\\Aloe_right_1.png")
        else:
            left_img = Image.open("data\\tsukuba\\left.png")
            right_img = Image.open("data\\tsukuba\\right.png")
    else:
        left_img = Image.open(left_img_uploader)
        right_img = Image.open(right_img_uploader)
    # display spinner
    with st.spinner('Wait for it...'):
        if method == "Pixel-wise":
            img = pixel_wise_matching(left_img, right_img, disparity_range=disparty_range, scale=scale, save_resutlts=False,
                                      distance_methods=compute_distance_l1)
        elif method == "Window-based":
            img = window_based_matching(left_img, right_img, disparity_range=disparty_range,
                                        kernel_size=kernel_size, save_resutlts=False)
        elif method == "Window-based with cosine similarity":
            img = window_based_matching_cosine(left_img, right_img, disparity_range=disparty_range,
                                               kernel_size=kernel_size, save_resutlts=False)

    st.markdown("### Results")
    st.image(img)
