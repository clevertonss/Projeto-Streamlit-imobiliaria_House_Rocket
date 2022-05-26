
# ------------------CEO gostaria de visualizar------------------------------------

# 1 - Filtros dos imoveis por um ou varias regioes.

#       - Objetivo: Visualizar as características dos imoveis.
#       - Acao do usuario: Digitar um ou mais codigos desejados.
#       - A visualizao: Uma tabela com todos os atributos selecionados.

# 2 - Escolher uma ou mais variaveis para visualizar.

#       - Objetivo: Visualizar imoveis por codigo postal.
#       - Acao do usuario: Digitar um ou mais codigos desejados.
#       - A visualizao: Uma tabela com todos os atributos e filtrada por codigo postal.

# 3 - Observar o numero total de imoveis, a media de preco, a media da sala de estar
        # e tambem a media do preco por metro quadrado em casa um dos codigos postais.

#       - Objetivo: Visualizar as medias de algumas métricas por região.
#       - Acao do usuario: Digita as métricas desejadas.
#       - A visualizao: Uma tabela com todos os atributos selecionados.

#
# 4 - Analisar casa uma das colunas de um modo mais descrito.

#       - Objetivo: Visualizar as métricas descritivas de cada atributo escolhido.
#       - Acao do usuario: Digita as métricas desejadas.
#       - A visualizao: Uma tabela com métricas descritivas por atributos.

# 5 - Um mapa com a densidade de portifolio por região e também desnsidade de preço.

#       - Objetivo: Visualizar a densidade de portfólio no mapa ( Número de imóveis por região )
#       - Acao do usuario: Nenhuma ação.
#       - A visualizao: Um mapa com a densidade de imoveis por regiao.

# 6 - Checar a variação anual de preço.

#       - Objetivo: Observar a variação anuais de preços.
#       - Acao do usuario: Filtrar os dados pelo ano.
#       - A visualizao: Um gráfico com linhas com os anos em X e preços médios em Y.

# 7 - Checar a variação diária de preço.

#       - Objetivo: Observar a variação diária.
#       - Acao do usuario: Filtrar os dados pelo ano.
#       - A visualizao: Um gráfico com linhas diárias em X e preços médios em Y

# 8 - Conferir a distribuição dos imóveis por:
#         - Preço
#         - Número de quartos.
#         - Número de banheiros.
#         - Número de andares.
#         - Vista para a água ou não.


# --------------------Libries------------------------------------------------------------

import folium
import geopandas
import streamlit      as st
import pandas         as pd
import numpy          as np
import plotly.express as px

from folium.plugins   import MarkerCluster
from streamlit_folium import folium_static
from datetime         import datetime, time

# ---------------------------------------------------------------------------------------
st.set_page_config( layout="wide") #Verificar o porque n'ao esta funcionando o codigo de centralizacao da page

st.title('House Rocket Company')
st.markdown( 'Welcome to House Rocket Data Analysis')
# st.header( 'Load data' )
# --------------------Read data----------------------------------------------------------

#Read data


@st.cache( allow_output_mutation=True )

def get_data( path ):
    data = pd.read_csv( path )
#     data['date'] = pd.to_datetime( data[ 'date' ] )

    return data

@st.cache( allow_output_mutation=True )
def get_geofile( url ):
    geofile = geopandas.read_file( url )

    return geofile

def set_feature (data ):

    # add new features
    data['price_m2'] = data['price'] / data['sqft_lot']

    return data

def overview_data( data ): 

    # =======================================================================================
    # Data Overview
    # =======================================================================================

    f_attributes = st.sidebar.multiselect( 'Enter Columns', sorted( set( data.columns ) ) )
    f_zipcode    = st.sidebar.multiselect( 'Enter Zipcode',sorted( set( data['zipcode'].unique() ) ) )

    st.header( 'Visão geral dos dados' )
    
    if( f_zipcode != [] ) & ( f_attributes != [] ):
        data = data.loc[ data[ 'zipcode' ].isin( f_zipcode ), f_attributes]

    elif( f_zipcode != [] ) & ( f_attributes == [] ):
        data = data.loc[ data[ 'zipcode' ].isin( f_zipcode ), : ]

    elif( f_zipcode == [] ) & ( f_attributes != [] ):
        data = data.loc[:, f_attributes ]

    else:
        data = data.copy()

    st.dataframe( data )


    c1, c2 = st.columns( (1, 1) )
    # Average metrics

    df1 = data[[ 'id', 'zipcode']]          .groupby( 'zipcode' ).count().reset_index()
    df2 = data[[ 'price', 'zipcode']]       .groupby( 'zipcode' ).mean().reset_index()
    df3 = data[[ 'sqft_living', 'zipcode' ]].groupby( 'zipcode' ).mean().reset_index()
    df4 = data[[ 'price_m2', 'zipcode' ]]   .groupby( 'zipcode' ).mean().reset_index()


    #merge
    m1 = pd.merge( df1, df2, on='zipcode', how='inner' )
    m2 = pd.merge( m1, df3, on='zipcode', how='inner' )
    df = pd.merge( m2, df4, on='zipcode', how='inner' )

    df.columns = ['ZIPCODE', 'TOTAL_HOUSES', 'PRICE', 'SQFT_LIVING', 'PRICE_M2' ]

    c1.header( 'Averege metrics' )
    c1.dataframe( df, height=200 )

    # Statistic Descriptive
    num_attributes = data.select_dtypes( include=['int64', 'float64' ] )
    media = pd.DataFrame( num_attributes.apply( np.mean ) )
    mediana = pd.DataFrame( num_attributes.apply( np.median) )
    std = pd.DataFrame( num_attributes.apply( np.std ) )

    max_ = pd.DataFrame( num_attributes.apply( np.max ) )
    min_ = pd.DataFrame( num_attributes.apply( np.min ) )

    df1 = pd.concat( [max_, min_, mediana, std, media ], axis=1 ).reset_index()

    df1.columns = ['attributes', 'max', 'min', 'mean', 'median', 'std' ]

    c2.header( 'Estatísca Descritiva' )
    c2.dataframe( df1, height=200 )

    return None

def portfolio_density( data, geofile ):
    # =======================================================================================
    # Densidade de portfólio
    # =======================================================================================
    # st.title( 'Region Overview' )

    c1, c2 = st.columns( ( 1, 1 ) )
    c1.header( 'Densidade do portfólio' )

    df = data

    # Base Map - folium

    # ---------------------------------------------------------------------------------------
    density_map = folium.Map(location=[data['lat'].mean(),
                            data['long'].mean() ],
                            default_zoom_start=15)

    marker_cluster = MarkerCluster().add_to( density_map )
    for name, row in df.iterrows():
        folium.Marker( [row['lat'], row['long'] ], 
            popup='Sold R${0} on: {1}. Features: {2} sqft, {3} bedrooms, {4} bathrooms, year built: {5}'.format( row['price'],
                                        row['date'],
                                        row['sqft_living'],
                                        row['bedrooms'],
                                        row['bathrooms'],
                                        row['yr_built'] ) ).add_to( marker_cluster )

    with c1:
        folium_static( density_map )

    # Region Price Map
    c2.header( 'Densidade de preço' )

    df = data[['price', 'zipcode']].groupby( 'zipcode' ).mean().reset_index()
    df.columns = ['ZIP', 'PRICE']

    # df = df.sample( 10 )

    geofile = geofile[geofile['ZIP'].isin( df['ZIP'].tolist() )]

    region_price_map = folium.Map( location=[data['lat'].mean(), 
                                data['long'].mean() ],
                                default_zoom_start=15 ) 


    region_price_map.choropleth( data = df,
                                geo_data = geofile,
                                columns=['ZIP', 'PRICE'],
                                key_on='feature.properties.ZIP',
                                fill_color='YlOrRd',
                                fill_opacity = 0.7,
                                line_opacity = 0.2,
                                legend_name='AVG PRICE' )

    with c2:
        folium_static( region_price_map)
    

    return None
    # ---------------------------------------------------------------------------------------
    # Plot map
    # st.write( 'House Rocket Map')
    # is_check = st.checkbox('Display Map')

    # ---------------------------------------------------------------------------------------

def commercial( data ):
    # ====================================================================================
    # Distribuição dos imóveis por categorias comerciais
    # ====================================================================================

    st.title(' Atributos comerciais' )
    st.sidebar.title( 'Opções comerciais' )

    # ------------- Média de preços por ano

    # Filtro
    min_year_built = int(data['yr_built'].min() )
    max_year_built = int(data['yr_built'].max() )

    st.header( 'Média de preços por ano de Construção' )
    st.sidebar.subheader( 'Selecione o ano de Construção' )

    f_year_built = st.sidebar.slider( 'Ano de Construção', min_year_built, max_year_built, min_year_built )


    # Data selectio
    df = data.loc[data[ 'yr_built' ] < f_year_built ]                       
    df = df[[ 'yr_built', 'price']].groupby( 'yr_built' ).mean().reset_index()

    # Plot
    fig = px.line( df, x='yr_built', y='price' )
    st.plotly_chart( fig, use_container_widht=True )

    # # ------------- Média preços por dia
    st.header( 'Média preços por dia' )
    st.sidebar.subheader( 'Selecione a data max' )

    # load data
    data = get_data( 'datasets/kc_house_data.csv' )
    data['date'] = pd.to_datetime( data['date'] ).dt.strftime( '%Y-%m-%d')

    # setup filters
    # 
    max_date = datetime.strptime( data['date'].max(), "%Y-%m-%d")
    min_date = datetime.strptime( data['date'].min(), "%Y-%m-%d" )
    f_date = st.sidebar.slider( 'date', min_date, max_date, min_date )

    # filter data
    # st.write( type( f_date ) )
    data['date'] = pd.to_datetime( data['date'] )
    df = data.loc[ data['date'] < f_date]
    df = df[['date', 'price']].groupby( 'date' ).mean().reset_index()
    fig = px.line( df, x='date', y='price' )
    st.plotly_chart( fig, use_container_width=True )

    # ------------------ Distribuição de preços
    # Histograma
    st.header( 'Distribuição de preço' )
    st.sidebar.subheader( 'Selecione o preço Máximo' )


    # Filter
    min_price = int( data['price'].min() )
    max_price = int( data['price'].max() )
    avg_price = int( data['price'].mean() )
    f_price = st.sidebar.slider('Price', min_price, max_price, avg_price )

    # Filter data
    df = data.loc[ data['price'] < f_price ]
    fig = px.histogram( df, x='price', nbins=50 )
    st.plotly_chart( fig, use_container_widht=True )

    return None

def commercial_distribution( data ):
    # ======================================================================
    # Dsitribução dos imóveis por categoria física
    # ======================================================================
    st.sidebar.title('Opções de atributos' )
    st.title('Atributos das casas' )

    # Filtro
    f_bedrooms   = st.sidebar.selectbox( 'Número máximo de quartos',    sorted( set( data['bedrooms'].unique() ) ) )
    f_bathrooms  = st.sidebar.selectbox( 'Número máximo de banheiros',  sorted( set( data['bathrooms'].unique() ) ) )
    f_floors     = st.sidebar.selectbox( 'Número máximo de andares',    sorted( set( data['floors'].unique() ) ) )
    f_waterfront = st.sidebar.checkbox( 'Número máximo de casas de frente para a água', sorted( set( data['waterfront'] ) ) )

    c1, c2 = st.columns( ( 1, 1 ) ) #Insere colunas, mas somente duas por x
    c3, c4 = st.columns( ( 1, 1 ) )

    # Casas por Quartos
    c1.header('Casas por quartos')
    df = data[ data['bedrooms'] < f_bedrooms ]
    fig = px.histogram( df, x='bedrooms', nbins=18 )
    c1.plotly_chart( fig, use_container_width=True )

    # Casas por Banheiros
    c2.header('Casas por banheiros' )
    df = data[ data['bathrooms']< f_bedrooms ]
    fig = px.histogram( data, x='bathrooms', nbins=25)
    c2.plotly_chart( fig, use_container_widht=True )

    # Casas por andar
    c3.header('Andares por casas')
    df = data.loc[ data['floors'] < f_floors ]
    fig = px.histogram( data, x='floors', nbins=20 )
    c3.plotly_chart( fig, use_container_width=True )

    # Casas de frente para água
    c4.header('Casas de frente para água' )
    df = data[ data['waterfront'] < f_waterfront ]
    fig = px.histogram( data, x='waterfront', nbins=20 )
    c4.plotly_chart( fig, use_container_width=True )
    
    return None

if __name__ == '__main__':
    # ETL
    data = get_data( '../kc_house_data.csv' )
    url = 'https://opendata.arcgis.com/datasets/83fc2e72903343aabff6de8cb445b81c_2.geojson'
    

    # load data
    data = get_data( path )
    geofile = get_geofile( url )

    # data extration


   
    # Transformation
    data = set_feature( data )
    
    overview_data( data )
    
    portfolio_density( data, geofile )
    
    commercial( data )                    #responsavel por criar os graficos de variacao
    
    commercial_distribution( data )




































