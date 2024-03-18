import streamlit as st
import torch                 as tc
import numpy                 as np
import matplotlib.pyplot     as plt
import pandas                as pd
from   torch                 import nn
from   torch.optim           import Adam
from   sklearn.preprocessing import MinMaxScaler
from   tqdm                  import tqdm
from sklearn import tree, model_selection, metrics

st.set_page_config(
    page_title="LSTM prediction",
    page_icon="🤖",
)

st.header('🤖 LSTM prediction ✨', divider='rainbow')

load_data, elab_data, training, test = st.tabs(["Caricamento dati", "Elaborazione dati", "Training", "Test"])

with load_data:

    uploaded_file = st.file_uploader('Carica qui il tuo dataset di train in formato json 👇', key=1)

    if uploaded_file is not None:
        tagaglobal_tr   = pd.read_json(uploaded_file)
        tagaglobal_test = tagaglobal_tr
        
        #st.help(tagaglobal_tr)
        st.subheader('Selezione serie temporale', divider='blue')
        genre = st.radio(
        "**Time series** 📊",
        ["Originale", "Rimescolata", "Originale + rimescolata"],
        horizontal=True,
        )

        st.write("Stai usando una time series: ", genre)
        if genre == "Rimescolata":
            tagaglobal_tr           = tagaglobal_tr.sample(frac=1)
            st.write(tagaglobal_tr)
        elif genre == "Originale + rimescolata":
            tagaglobal_tr_rim       = tagaglobal_tr
            tagaglobal_tr_rim       = tagaglobal_tr_rim.sample(frac=1)
            tagaglobal_tr_new       = tagaglobal_tr._append(tagaglobal_tr_rim, ignore_index=True)
            tagaglobal_tr =         tagaglobal_tr_new
            st.write(tagaglobal_tr)
        else: 
            st.write(tagaglobal_tr) 
        
            
        with elab_data:
            sel_periodo, normalizzazione, tensore = st.columns(3)
            
            with sel_periodo:
                # CONSIDERIAMO LE COLONNE PRODOTTO E MATTINA
                st.subheader('1) Selezione periodo', divider='blue')
                tagaglobal_tr_sel_col = st.multiselect('Seleziona colonne:',  tagaglobal_tr.columns)
                if len(tagaglobal_tr_sel_col) > 0:
                    tagaglobal_tr_sel = tagaglobal_tr[tagaglobal_tr_sel_col]
                    tagaglobal_tr_sel
                    
                    with normalizzazione:
                        st.subheader('2) Normalizzazione', divider='blue')
                        #st.subheader('Consideriamo le colonne PRODOTTO e MATTINA e normalizzazione dati MATTINA (0, 1)')
                        #tagaglobal_tr_sel    = tagaglobal_tr[["PRODOTTO", "MATTINA"]]
                        min_max_scalar       = MinMaxScaler(feature_range=(0,1))
                        tagaglobal_tr_sel_nor_col = st.selectbox('Normalizza colonna:', tagaglobal_tr_sel_col)
                        if len(tagaglobal_tr_sel_nor_col) > 0:
                            tagaglobal_tr_sel_nor = tagaglobal_tr_sel[tagaglobal_tr_sel_nor_col]
                            #tagaglobal_tr_sel_nor
                            #tagaglobal_tr_sel_nor = st.selectbox('Normalizza colonna', tagaglobal_tr_sel)
                            #tagaglobal_tr_scaled = min_max_scalar.fit_transform(tagaglobal_tr_sel[["MATTINA"]])
                            tagaglobal_tr_scaled = min_max_scalar.fit_transform(tagaglobal_tr_sel[[tagaglobal_tr_sel_nor_col]])
                            st.write(tagaglobal_tr_scaled)

                            with tensore:
                                if len(tagaglobal_tr_scaled) > 0:
                                    # TRAFORMAZIONE IN TENSORE
                                    st.subheader('3) Trasformazione in tensore', divider='blue')
                                    tagaglobal_tr_tensor = tc.FloatTensor(tagaglobal_tr_scaled)
                                    
                                    if st.toggle('Mostra tensore'):
                                        st.code(tagaglobal_tr_tensor)
                                    
                                    #st.text(tagaglobal_tr_tensor)
                                    
                                    # SUDDIVIZIONE SERIE TEMPORALE IN FINESTRE DA "train_window" ELEMENTI
                                    st.subheader('4) Suddivisione in finestre', divider='blue')
                                    train_window = st.slider('Seleziona la dimensione delle finestre in cui ridurre la serie temporale:', value=15, min_value=1, max_value=60)

                                    def create_inout_sequences(input_data, tw):
                                        inout_seq = []
                                        input_len = len(input_data)
                                        for i in range(input_len - tw):
                                            train_seq   = input_data[i:i+tw]
                                            train_label = input_data[i+tw:i+tw+1]
                                            inout_seq.append((train_seq ,train_label))
                                        return inout_seq

                                    #train_window    = 15
                                    train_inout_seq = create_inout_sequences(tagaglobal_tr_tensor, train_window)
                                    st.subheader(':rainbow[Finestre ottenute]')
                                    st.write(train_inout_seq)
                                    
                                    class LSTM(nn.Module):
                                        def __init__(self, device_obj, input_size=1, hidden_layer_size=100, output_size=1, layers_number=1, dropout=0.2):
                                            super().__init__()
                                            self.hidden_layer_size = hidden_layer_size
                                            self.layers_number     = layers_number
                                            self.device_obj        = device_obj

                                            self.lstm              = nn.LSTM(input_size, hidden_layer_size, layers_number, dropout=dropout)
                                            self.linear            = nn.Linear(hidden_layer_size, output_size)

                                            self.hidden_cell       = (
                                                tc.randn(layers_number, input_size, hidden_layer_size).to(self.device_obj),
                                                tc.randn(layers_number, input_size, hidden_layer_size).to(self.device_obj)
                                            )

                                        def forward(self, batch):
                                            batch_size            = len(batch)
                                            lstm_input            = batch.view(batch_size ,1, -1).to(self.device_obj)
                                            lstm_out, hidden_cell = self.lstm(lstm_input, self.hidden_cell)
                                            self.hidden_cell      = (
                                                hidden_cell[0].detach().to(self.device_obj),
                                                hidden_cell[1].detach().to(self.device_obj)
                                            )
                                            predictions           = self.linear(lstm_out.view(len(batch), -1))
                                            return predictions[-1]
                                    
                                    ## CREAZIONE MODELLO, CARICAMENTO DATI IN GPU
                                    def to_device(element, device_obj):
                                        if isinstance(element, (tuple, list)):
                                            return [
                                            to_device(sub_elem, device_obj)
                                            for sub_elem in element
                                        ]
                                        return element.to(device_obj)

                                    st.sidebar.markdown("# 🤖 :rainbow[LSTM prediction] ✨")
                                    st.sidebar.markdown("# 🧠 :blue[Model parameters] 🪄")
                                    
                                    p_input_size        = st.sidebar.slider("Seleziona il numero delle features che compongono l'**:red[input]**:", value=1, min_value=1, max_value=100)
                                    p_output_size       = st.sidebar.slider("Seleziona il numero delle features in **:red[output]**:", value=1, min_value=1, max_value=75)
                                    p_hidden_layer_size = st.sidebar.slider("Seleziona il numero di **:red[neuroni nei livelli intermedi]**:", value=100, min_value=1, max_value=1000)
                                    p_layers_number     = st.sidebar.slider("Seleziona il numero di **:red[livelli]**:", value=1, min_value=1, max_value=1000)
                                    p_device            = st.sidebar.selectbox('Seleziona il **:red[device]**:', ['cpu','cuda'])
                                    p_epochs            = st.sidebar.slider("Seleziona il numero di **:red[epoche]**:", value=8, min_value=1, max_value=100)
                                    
                                    #device          = tc.device('cuda') if tc.cuda.is_available() else tc.device("cpu")
                                    device          = tc.device(p_device)
                                    model           = LSTM(device, input_size=p_input_size, hidden_layer_size=p_hidden_layer_size, output_size=p_output_size, layers_number=p_layers_number)
                                    loss_function   = nn.MSELoss()
                                    optimizer       = Adam(model.parameters(), lr=0.001)

                                    model.to(device)
                                    train_inout_seq = to_device(train_inout_seq, device)
                        
                                    ## TRAINING
                                    def model_training(eph, input_data, ls_fn, opt):
                                        losses     = []
                                        for i in range(eph):
                                            losses_i = []
                                            for seq, label in tqdm(input_data):
                                                # PREDIZIONE
                                                y_pred = model(seq)

                                                # CALCOLO LOSS
                                                single_loss = ls_fn(y_pred, label)
                                                losses_i.append(single_loss)

                                                # CALCOLO GRADIENTE
                                                opt.zero_grad()
                                                single_loss.backward(retain_graph=True)
                                                opt.step()

                                            losses.append(tc.stack(losses_i).mean().item())
                                        return losses

                                    with training:
                                        st.subheader('Training del modello', divider='blue')
                                        epochs        = p_epochs
                                        if st.button('🚀 Esegui il training 💡'):
                                            # Show a spinner during a process
                                            with st.spinner(text='Training in progress. ⏳ Attendere prego...'):
                                                computed_loss = model_training(epochs, train_inout_seq, loss_function, optimizer)
                                                st.success('💪🏻Training completato! 🎉')
                            
                                            st.subheader('🏆 Risultati training', divider='rainbow')
                                            ## STAMPARE LOSS TRAIN
                                            fig    = plt.figure(figsize=(10,5))
                                            x_axis = list(range(1, epochs + 1))
                                            #plt.plot(x_axis, computed_loss)
                                            #plt.show()
                                            chart_data = pd.DataFrame({"loss":computed_loss, "epoche":x_axis}) # , columns=['epoche','loss']
                                            chart_data
                                            st.line_chart(data=chart_data, x="epoche", y="loss", color=["#0000FF"], width=epochs) #, color=["#FF0000", "#0000FF"]
                                            
                                            st.balloons()
        with test:
            st.subheader('1) Selezione serie temporale', divider='blue')
            #uploaded_file_test = st.file_uploader('Carica qui il tuo dataset di test in formato json 👇', key=2)
            #tagaglobal_test
                
            genre_test = st.radio(
            "**Time series di test** 📊",
            ["Originale", "Rimescolata", "Originale + rimescolata"],
            horizontal=True,
            key=2,
            )

            st.write("Stai usando una time series: ", genre_test)
            if genre_test == "Rimescolata":
                tagaglobal_test           = tagaglobal_test.sample(frac=1)
                st.write(tagaglobal_test)
            elif genre_test == "Originale + rimescolata":
                tagaglobal_test_rim       = tagaglobal_test
                tagaglobal_test_rim       = tagaglobal_test_rim.sample(frac=1)
                tagaglobal_test_new       = tagaglobal_test._append(tagaglobal_test_rim, ignore_index=True)
                tagaglobal_test           = tagaglobal_test_new
                st.write(tagaglobal_test)
            else: 
                st.write(tagaglobal_test) 
            
            sel_periodo_test, normalizzazione_test, tensore_test = st.columns(3)
            
            with sel_periodo_test:
                # CONSIDERIAMO LE COLONNE PRODOTTO E MATTINA
                st.subheader('2) Selezione periodo', divider='blue')
                tagaglobal_test_sel_col = st.multiselect('Seleziona colonne per il test:',  tagaglobal_test.columns, key=3)
                if len(tagaglobal_test_sel_col) > 0:
                    tagaglobal_test_sel = tagaglobal_test[tagaglobal_test_sel_col]
                    tagaglobal_test_sel
                    
                    with normalizzazione_test:
                        st.subheader('3) Normalizzazione', divider='blue')
                        #st.subheader('Consideriamo le colonne PRODOTTO e MATTINA e normalizzazione dati MATTINA (0, 1)')
                        #tagaglobal_tr_sel    = tagaglobal_tr[["PRODOTTO", "MATTINA"]]
                        min_max_scalar_test       = MinMaxScaler(feature_range=(0,1))
                        tagaglobal_test_sel_nor_col = st.selectbox('Normalizza colonna per il test:', tagaglobal_test_sel_col)
                        if len(tagaglobal_test_sel_nor_col) > 0:
                            tagaglobal_test_sel_nor = tagaglobal_test_sel[tagaglobal_test_sel_nor_col]
                            #tagaglobal_tr_sel_nor
                            #tagaglobal_tr_sel_nor = st.selectbox('Normalizza colonna', tagaglobal_tr_sel)
                            #tagaglobal_tr_scaled = min_max_scalar.fit_transform(tagaglobal_tr_sel[["MATTINA"]])
                            tagaglobal_test_scaled = min_max_scalar_test.fit_transform(tagaglobal_test_sel[[tagaglobal_test_sel_nor_col]])
                            st.write(tagaglobal_test_scaled)

                            with tensore_test:
                                if len(tagaglobal_test_scaled) > 0:
                                    # TRAFORMAZIONE IN TENSORE
                                    st.subheader('4) Trasformazione in tensore', divider='blue')
                                    tagaglobal_test_tensor = tc.FloatTensor(tagaglobal_test_scaled)
                                    
                                    if st.toggle('Mostra tensore', key=4):
                                        st.code(tagaglobal_test_tensor)
                                    
                                    #st.text(tagaglobal_tr_tensor)
                                    
                                    # SUDDIVIZIONE SERIE TEMPORALE IN FINESTRE DA "train_window" ELEMENTI
                                    st.subheader('5) Suddivisione in finestre', divider='blue')
                                    test_window = st.slider('Seleziona la dimensione delle finestre in cui ridurre la serie temporale di test:', value=15, min_value=1, max_value=60, key=5)

                                    test_inout_seq        = create_inout_sequences(tagaglobal_test_tensor, test_window)
                                    test_inout_seq        = to_device(test_inout_seq, device)
                                    st.subheader(':rainbow[Finestre ottenute]')
                                    st.write(test_inout_seq)
                                    
            st.subheader('6) Test del modello', divider='blue')
            if st.button('🚀 Esegui il test 🧪', key=6):
                # Show a spinner during a process
                with st.spinner(text='Test in progress. ⏳ Attendere prego...'):
                    pred = []
                    real = []
                    for seq, label in tqdm(test_inout_seq):
                        # PREDIZIONE
                        y_pred = model(seq)
                        pred.append(y_pred.item())
                        real.append(label.item())

                st.success('Test completato! 🎉')

                st.subheader('🏆 Risultati test', divider='rainbow')
                pred_res = min_max_scalar.inverse_transform([pred])
                real_res = min_max_scalar.inverse_transform([real])
                print(pred_res.tolist())
                print(real_res.tolist())
                chart_data_test = pd.DataFrame({"Predictions":pred_res[0], "Actual":real_res[0]}) # , columns=['epoche','loss']
                chart_data_test
                st.line_chart(data=chart_data_test, color=["#0000FF", "#4c974c"]) #, color=["#FF0000", "#0000FF"]
                
                mse = metrics.mean_squared_error(real_res, pred_res)
                st.write("mse: ", mse) 
                
                st.balloons()
                
                
                            
    