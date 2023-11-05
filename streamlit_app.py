import streamlit as st

from src.model.predictor import predict


def main():
    st.set_page_config(page_title="Head Detection")
    st.title("Face Detection")

    st.sidebar.title("Settings")
    image = st.sidebar.file_uploader("Face", label_visibility="collapsed")

    conf_slider = st.sidebar.slider("Confidence", min_value=0.0, max_value=1.0, value=0.5)
    iou_slider = st.sidebar.slider("Intersection", min_value=0.0, max_value=1.0, value=0.7)

    if st.sidebar.button("Predict"):
        if image is not None:
            result = predict(image.read(), conf=conf_slider, iou=iou_slider)

            st.image(result, channels="BGR")
        else:
            st.write(":red[Please upload an image first!]")


if __name__ == "__main__":
    main()
