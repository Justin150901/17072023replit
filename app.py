import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model
import yfinance as yf
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

def format_rupiah(amount):
    return 'Rp {:,.2f}'.format(amount)

def sequences(data, seq_length=60):
    X = []
    y = []
    for i in range(seq_length, len(data)):
        X.append(data[i-seq_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)

def forecast(model, input_data, num_units, units_type, seq_length=60):
    steps = 365 * num_units

    future_predictions = []
    current_input = input_data[-seq_length:].reshape(1, seq_length, 1)

    for _ in range(steps):
        prediction = model.predict(current_input)[0][0]
        future_predictions.append(prediction)
        current_input = np.roll(current_input, -1)
        current_input[0, -1, 0] = prediction

    return future_predictions

def main():
    st.set_page_config(page_title="Sahamin Aja", layout="centered")
    st.image("Frame 1.png")
    st.title("Aplikasi Prediksi Harga Saham Perbankan")
    st.write("")
    st.write("Selamat Datang Pada Aplikasi Prediksi Harga Saham.")
    st.write("Silahkan Pilih Saham yang Ingin dilakukan Prediksi.")

    saham = {
        'BBCA.JK' : 'BBCA.JK (PT Bank Central Asia Tbk)',
        'AGRO.JK' : 'AGRO.JK (PT Bank Jago Tbk)',
        'AGRS.JK' : 'AGRS.JK (PT Bank Agris Tbk)',
        'AMAR.JK' : 'AMAR.JK (PT Bank Amar Indonesia Tbk)',
        'ARTO.JK' : 'ARTO.JK (PT Bank Artos Indonesia Tbk)',
        'BABP.JK' : 'BABP.JK (PT Bank MNC Internasional Tbk)',
        'BACA.JK' : 'BACA.JK (PT Bank Capital Indonesia Tbk)',
        'BANK.JK' : 'BANK.JK (PT Bank of India Indonesia Tbk)',
        'BBKP.JK' : 'BBKP.JK (PT Bank KB Bukopin Tbk)',
        'BBMD.JK' : 'BBMD.JK (PT Bank Mandom Indonesia Tbk)',
        'BBNI.JK' : 'BBNI.JK (PT Bank Negara Indonesia (Persero) Tbk)',
        'BBRI.JK' : 'BBRI.JK (PT Bank Rakyat Indonesia (Persero) Tbk)',
        'BBSI.JK' : 'BBSI.JK (PT Bank Bisnis Internasional Tbk)',
        'BBTN.JK' : 'BBTN.JK (PT Bank Tabungan Negara (Persero) Tbk)',
        'BBYB.JK' : 'BBYB.JK (PT Bank Yudha Bhakti Tbk)',
        'BCIC.JK' : 'BCIC.JK (PT Bank Jasa Jakarta Tbk)',
        'BDMN.JK' : 'BDMN.JK (PT Bank Danamon Indonesia Tbk)',
        'BEKS.JK' : 'BEKS.JK (PT Bank Pembangunan Daerah Banten Tbk)',
        'BGTG.JK' : 'BGTG.JK (PT Bank Bintang Tbk)',
        'BINA.JK' : 'BINA.JK (PT Bank Ina Perdana Tbk)',
        'BJBR.JK' : 'BJBR.JK (PT Bank Pembangunan Daerah Jawa Barat dan Banten Tbk)',
        'BJTM.JK' : 'BJTM.JK (PT Bank Pembangunan Daerah Jawa Tengah)',
        'BKSW.JK' : 'BKSW.JK (PT Bank Sahabat Sampoerna Tbk)',
        'BMAS.JK' : 'BMAS.JK (PT Bank Multi Arta Sentosa Tbk)',
        'BMRI.JK' : 'BMRI.JK (PT Bank Mandiri (Persero) Tbk)',
        'BNBA.JK' : 'BNBA.JK (PT Bank Bumi Arta Tbk)',
        'BNGA.JK' : 'BNGA.JK (PT Bank CIMB Niaga Tbk)',
        'BNII.JK' : 'BNII.JK (PT Bank Maybank Indonesia Tbk)',
        'BNLI.JK' : 'BNLI.JK (PT Bank Permata Tbk)',
        'BRIS.JK' : 'BRIS.JK (PT Bank BRIsyariah Tbk)',
        'BSIM.JK' : 'BSIM.JK (PT Bank Sinarmas Tbk)',
        'BTPN.JK' : 'BTPN.JK (PT Bank BTPN Tbk)',
        'BTPS.JK' : 'BTPS.JK (PT Bank Tabungan Pensiunan Nasional Syariah Tbk)',
        'BVIC.JK' : 'BVIC.JK (PT Bank Victoria International Tbk)',
        'DNAR.JK' : 'DNAR.JK (PT Bank Dinar Indonesia Tbk)',
        'INPC.JK' : 'INPC.JK (PT Bank Artha Niaga Kencana Tbk)',
        'MASB.JK' : 'MASB.JK (PT Bank Maspion Indonesia Tbk)',
        'MAYA.JK' : 'MAYA.JK (PT Bank Mayapada Internasional Tbk)',
        'MCOR.JK' : 'MCOR.JK (PT Bank China Construction Bank Indonesia Tbk)',
        'MEGA.JK' : 'MEGA.JK (PT Bank Mega Tbk)',
        'NISP.JK' : 'NISP.JK (PT Bank OCBC NISP Tbk)',
        'NOBU.JK' : 'NOBU.JK (PT Bank Nationalnobu Tbk)',
        'PNBN.JK' : 'PNBN.JK (PT Bank Pan Indonesia Tbk)',
        'PNBS.JK' : 'PNBS.JK (PT Bank Panin Dubai Syariah Tbk)',
        'SDRA.JK' : 'SDRA.JK (PT Bank Harda Internasional Tbk)'
    } 
    

    st.write("")
    pilihan_saham = st.selectbox("Pilih Saham:", list(saham.values()))
    ticker_saham = pilihan_saham.split(' ')[0]

    start_date = pd.to_datetime("2017-01-01")
    end_date = pd.to_datetime("2021-12-31")

    df = yf.download(ticker_saham, start=start_date, end=end_date)
    df.dropna(inplace=True)

    st.write("Data Saham Dari 1 Januari 2017 - 31 Desember 2021")
    st.dataframe(df.style.highlight_max(axis=0))

    st.write("Grafik Saham Dari 1 Januari 2017 - 31 Desember 2021")
    st.line_chart(df['Adj Close'])

    jumlah_lot = st.number_input("Masukkan Jumlah Lot:", value=1, step=1)

    harga_saham_terakhir = df['Adj Close'][-1]
    estimasi_biaya = harga_saham_terakhir * jumlah_lot * 100  
    st.write(f"Estimasi biaya untuk membeli {jumlah_lot} lot saham adalah: {format_rupiah(estimasi_biaya)}")
    st.write("")

    units_type = "Tahun"
    num_units = 1 

    if st.button("Lakukan Prediksi"):
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df['Adj Close'].values.reshape(-1,1))

        model_path = f"{ticker_saham.replace('.JK', '')}.h5"
            
        if not os.path.isfile(model_path):
            st.error(f"Model Tidak Ditemukan {ticker_saham}. Harap pastikan file model {model_path} itu ada.")
            return
        
        model = load_model(model_path)

        future_predictions = forecast(model, scaled_data, num_units, units_type)
        future_predictions = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1))

        future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=len(future_predictions), freq='D')
        future_df = pd.DataFrame(data=future_predictions, index=future_dates, columns=['Predictions'])

        st.write("")
        st.write("")
        st.write(f"Harga Prediksi Menunjukkan: {format_rupiah(future_predictions[-1][0])} pada tanggal {future_dates[-1].strftime('%Y-%m-%d')}")

        st.write("Grafik Saham Setelah Dilakukan Prediksi")
        chart_data = pd.concat([df['Adj Close'], future_df['Predictions']], axis=1)
        st.line_chart(chart_data)

        fig, ax = plt.subplots(figsize=(10, 5))

        chart_data.plot(ax=ax)
        ax.set_ylabel('Close Price')

        for year in range(start_date.year, end_date.year + 2):
            ax.axvline(pd.to_datetime(f'{year}-01-01'), color='red', linestyle='--', lw=1)

        st.write("Grafik Saham Setelah Dilakukan Prediksi (dengan garis vertikal di awal tahun)")
        st.pyplot(fig)

        harga_awal = df['Adj Close'][-1]
        harga_akhir = future_predictions[-1][0]

        keuntungan_per_saham = harga_akhir - harga_awal
        keuntungan_total = keuntungan_per_saham * jumlah_lot * 100  

        investasi_awal = harga_awal * jumlah_lot * 100
        persentase_keuntungan = (keuntungan_total / investasi_awal) * 100
        keuntungan_total_modal = keuntungan_total + estimasi_biaya
        estimasi_keuntungan_perbulan = keuntungan_total / 12

        st.write("")
        st.write(f"Jika Anda membeli {jumlah_lot} lot {ticker_saham} pada tanggal {df.index[-1].strftime('%Y-%m-%d')} maka pada tanggal {future_dates[-1].strftime('%Y-%m-%d')}")
        st.write(f"Anda akan mendapatkan keuntungan {format_rupiah(keuntungan_total)}")
        st.write("")
        
        st.markdown("""
        | Keterangan | Nilai |
        | --- | --- |
        | Modal Berdasarkan Lot | {} |
        | Keuntungan Bersih Per Tahun | {} |
        | Estimasi Keuntungan Per Bulan | {} |
        | Keuntungan Total (Modal + Bersih) | {} |
        | Kenaikan Modal | {:.2f}% |
        """.format(format_rupiah(estimasi_biaya), format_rupiah(keuntungan_total), format_rupiah(estimasi_keuntungan_perbulan), format_rupiah(keuntungan_total_modal), persentase_keuntungan))
       
        st.markdown("""
        ***
        """)
        st.markdown("""
        <p class="small">Copyright © 2023 | All Rights Reserved | Justin Septianto Sutjipta | Version 1.11.0</p>
        """, unsafe_allow_html=True)
if __name__ == "__main__":
    main()
