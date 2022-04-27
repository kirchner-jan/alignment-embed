import streamlit as st
import umap , umap.plot , pickle
import numpy as np

st.title('UMAP embedding of literature on AI Alignment')

embed_dict = pickle.load( open( "uber-embed.p", "rb" ) )
all_embeddings = np.array([x['embedding'] for x in embed_dict.values()])

cvec = pickle.load( open( "cvec.p", "rb" ) )
mapper = umap.UMAP(n_neighbors=50).fit(all_embeddings[np.where(cvec%2 != 1)[0],:])
hover_data = pickle.load( open( "hover_data.p", "rb" ) )

umap.plot.output_notebook()
p = umap.plot.interactive(mapper, labels=cvec[np.where(cvec%2 != 1)[0]], hover_data=hover_data, point_size=10, theme='fire')
# umap.plot.show(p)
st.bokeh_chart(p, use_container_width=True)
