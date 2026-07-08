"""Text-based visualizations. Opt-in: requires `wordcloud`."""
import matplotlib.pyplot as plt


def plot_wordcloud(text, title, colormap="viridis"):
    from wordcloud import WordCloud

    if not text.strip() or len(text) < 10:
        print(f"Not enough data to generate WordCloud for: {title}")
        return

    wc = WordCloud(width=800, height=400, background_color="white", colormap=colormap).generate(text)
    plt.figure(figsize=(12, 6))
    plt.imshow(wc, interpolation="bilinear")
    plt.axis("off")
    plt.title(title, fontsize=20, pad=20)
    plt.show()
