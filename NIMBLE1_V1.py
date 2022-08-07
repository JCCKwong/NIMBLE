"""
Title: Non-muscle invasive bladder cancer longitudinal evaluation Phase 1 (NIMBLE-1): Initial diagnosis
"""
# Import packages and libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
from sksurv.ensemble import RandomSurvivalForest
import joblib

def main():
    if "page" not in st.session_state:
        # Initialize session state.
        st.session_state.update({
            # Default page.
            "page": "NIMBLE-1"
        })

    st.sidebar.image("images/Logo.png", use_column_width=True)

    page = st.sidebar.radio("", tuple(PAGES.keys()))

    PAGES[page]()

def page_nimble1():
    st.title("NIMBLE-1 (Non-Muscle Invasive Bladder Cancer Longitudinal Evaluation: Initial Diagnosis)")
    st.sidebar.header("Instructions")
    st.sidebar.markdown(
        """
    1. Enter your information on the right
    1. Press the SUBMIT button
    1. NIMBLE-1 will output the following:
        * Probability of avoiding tumour progression at 1 and 5 years from initial diagnosis
        * Individualized progression-free survival curve
    """
    )

    # Specify font size for annotated prostate diagram
    #font = ImageFont.truetype('Images/Font.ttf', 80)

    # Load saved items from Google Drive
    model = joblib.load('model/NIMBLE-1.pkl')

    # Input individual values in sidebar

    col1, col2 = st.columns([1, 1])
    col1.header("Enter Your Information")
    if col1.button('Click here if you only have carcinoma in situ (CIS) from your surgery (TURBT)'):
        col1.warning('The results of NIMBLE-1 will not be applicable for you since this tool was only trained on '
                     'patients who also had Ta or T1 disease.')
    col2.header("Risk of Tumour Progression")
    col2.write("This graph displays your personalized risk of being free from progression to muscle-invasive bladder "
               "cancer over time.")

    choices = {'Male': 0,
               'Female': 1,
               'Ta (Non-invasive papillary carcinoma)': 0,
               'T1 (Tumour infiltrating the lamina propria)': 1,
               'No': 0,
               'Yes': 1,
               'Grade 1': 1,
               'Grade 2': 2,
               'Grade 3': 3,
               'Single tumour': 0,
               'Multiple tumours': 1,
               'Less than 3 cm': 0,
               '3 cm or greater': 1

               }

    age = col1.number_input("Age (years)", 0, 100, 72)
    sex = col1.radio("Sex", options=('Male', 'Female'), horizontal=True, index=0)
    stage = col1.radio("Tumour Stage", options=('Ta (Non-invasive papillary carcinoma)',
                                                'T1 (Tumour infiltrating the lamina propria)'), horizontal=True,
                       index=1)
    cis = col1.radio("Concomittant CIS", options=('No', 'Yes'), horizontal=True, index=1)
    grade = col1.radio("Tumour Grade (WHO 1973)", options=('Grade 1', 'Grade 2', 'Grade 3'), horizontal=True,
                       index=2)
    number = col1.radio("Number of Tumours", options=('Single tumour', 'Multiple tumours'), horizontal=True,
                        index=0)
    size = col1.radio("Tumour Size", options=('Less than 3 cm', '3 cm or greater'), horizontal=True, index=1)
    bcg = col1.radio("Treated with Intravesical Bacillus Calmette-Guérin (BCG)", options=('No', 'Yes'),
                     horizontal=True, index=0)
    mmc = col1.radio("Treated with Intravesical Mitomycin C (MMC)", options=('No', 'Yes'), horizontal=True, index=0)

    ### DATA STORAGE ###
    sex = choices[sex]
    stage = choices[stage]
    cis = choices[cis]
    grade = choices[grade]
    number = choices[number]
    size = choices[size]
    bcg = choices[bcg]
    mmc = choices[mmc]

    def calculatescore(sex, age, number, size, stage, cis, grade):
        EORTC_rec = 0
        EORTC_prog = 0
        CUETO_rec = 0
        CUETO_prog = 0
        EAU_prog = 0
        CUA_risk = 0

        # gender
        CUETO_prog += 0
        CUETO_rec += (sex == 1) * 3

        # age
        CUETO_prog += (age > 70) * 2
        CUETO_rec += (age >= 60) + (age > 70)
        EAU_prog += (age > 70) * 32

        # number of tumors
        EORTC_prog += number * 3
        EORTC_rec += number * 3
        CUETO_prog += number
        CUETO_rec += number * 2
        EAU_prog += number * 32
        CUA_risk += number

        # tumor diameter
        EORTC_prog += size * 3
        EORTC_rec += size * 3
        EAU_prog += size * 43
        CUA_risk += size

        # stage
        EORTC_prog += stage * 4
        EORTC_rec += stage
        CUETO_prog += stage * 2
        CUETO_rec += 0
        EAU_prog += stage * 52

        # concurrent cis
        EORTC_prog += cis * 6
        EORTC_rec += cis
        CUETO_prog += cis
        CUETO_rec += cis * 2
        EAU_prog += cis * 58

        # grade
        EORTC_prog += (grade == 3) * 5
        EORTC_rec += grade - 1
        CUETO_prog += (grade == 2) * 2 + (grade == 3) * 6
        CUETO_rec += (grade == 2) + (grade == 3) * 3
        EAU_prog += (grade == 2) * 58 + (grade == 3) * 100

        # EAU progression ratio
        EAU_prog_ratio = EAU_prog/317

        # CUA risk group
        if (number == 0) & (size == 0) & (stage == 0) & (cis == 0) & (grade < 3):
            CUA_score = 0

        elif (stage == 1) & (grade == 3) & (((number == 1) & (size == 1)) or (cis == 1)):
            CUA_score = 5

        elif (stage > 0) or ((stage == 0) & (grade == 3) & ((number == 1) or (size == 1))):
            CUA_score = 4

        elif (stage == 0) & (grade == 3) & (number == 0) & (size == 0):
            CUA_score = 3

        elif (CUA_risk > 0):
            CUA_score = 2

        else:
            CUA_score = 1

        return [EORTC_rec, EORTC_prog, CUETO_rec, CUETO_prog, EAU_prog_ratio, CUA_score]

    EORTC_rec, EORTC_prog, CUETO_rec, CUETO_prog, EAU_prog_ratio, CUA_score = calculatescore(sex,
                                                                                             age,
                                                                                             number,
                                                                                             size,
                                                                                             stage,
                                                                                             cis,
                                                                                             grade)

    # Store a dictionary into a variable
    pt_data = {'Age at Initial Diagnosis': age,
               'Tumour Grade': grade,
               'Treated with BCG': bcg,
               'Treated with MMC': mmc,
               'EORTC - Recurrence Score': EORTC_rec,
               'EORTC - Progression Score': EORTC_prog,
               'CUETO - Recurrence Score': CUETO_rec,
               'CUETO - Progression Score': CUETO_prog,
               'EAU - Progression Ratio': EAU_prog_ratio,
               'CUA - Risk Group': CUA_score
               }

    pt_features = pd.DataFrame(pt_data, index=[0])

    survival = model.predict_survival_function(pt_features)

    # Display the survival function
    plt.rcParams.update({'text.color': 'white',
                         'axes.labelcolor': 'white',
                         'xtick.color': 'white',
                         'ytick.color': 'white'})

    fig, ax = plt.subplots()
    for fn in survival:
        plt.step(fn.x, fn(fn.x), where="post", color='blue', lw=2, ls='-')

    # Axis labels
    plt.xlabel('Time from initial diagnosis (years)')
    plt.ylabel('Progression-free survival (%)')

    # Tick labels
    plt.ylim(0, 1.05)
    y_positions = (0, 0.2, 0.4, 0.6, 0.8, 1)
    y_labels = ('0', '20', '40', '60', '80', '100')
    plt.yticks(y_positions, y_labels, rotation=0)
    plt.xlim(0, 10)
    plt.xticks(np.arange(0, 11, step=1))

    # Tick vertical lines
    plt.axvline(x=1, color='black', ls='--', alpha=0.2)
    plt.axvline(x=5, color='black', ls='--', alpha=0.2)

    fig.patch.set_facecolor('#0E1117')
    ax.patch.set_facecolor('white')

    # Calculate probability of progression-free survival at 1 and 5 years
    survival_1 = str(np.round(np.interp(1, fn.x, fn(fn.x)) * 100, 1))
    survival_5 = str(np.round(np.interp(5, fn.x, fn(fn.x)) * 100, 1))

    # Display survival curve
    col2.pyplot(fig)
    col2.write('**Probability of avoiding tumour progression at 1 year: {}%**'.format(survival_1))
    col2.write('**Probability of avoiding tumour progression at 5 years: {}%**'.format(survival_5))

    # Download variables at initial diagnosis
    download_data = {'Age at Initial Diagnosis': age,
                     'Sex': sex,
                     'Tumour Stage': stage,
                     'Concomittant CIS': cis,
                     'Tumour Grade (WHO 1973)': grade,
                     'Number of Tumours': number,
                     'Tumour Diameter': size,
                     'Treated with BCG': bcg,
                     'Treated with MMC': mmc,
                     'Progression-free survival at 1 year (%)': survival_1,
                     'Progression-free survival at 5 years (%)': survival_5
                     }

    download_data = pd.DataFrame(download_data, index=[0]).to_csv()

    col2.download_button(label="Download data to use in NIMBLE-2 (Longitudinal Evaluation)",
                         data=download_data,
                         file_name='initial_diagnosis.csv',
                         mime='text/csv',
                         )

def page_nimble2():
    st.title("NIMBLE-2 (Non-Muscle Invasive Bladder Cancer Longitudinal Evaluation: Follow-up)")
    st.file_uploader(label="Import Initial Diagnosis Info from NIMBLE-1 (.csv file only)",
                     type="csv")
    st.write("Under development")

def page_about():
    st.title("NIMBLE (Non-Muscle Invasive Bladder Cancer Longitudinal Evaluation)")
    st.markdown(
        """
    Welcome to the Non-Muscle Invasive Bladder Cancer Longitudinal Evaluation (NIMBLE) tool. NIMBLE is divided into 
    two components: NIMBLE-1 (Initial Diagnosis) and NIMBLE-2 (Follow-up). Both tools provide personalized dynamic 
    prognostication for tumour progression in non-muscle invasive bladder cancer patients. \n
    NIMBLE-1 provides several outputs that may be beneficial for treatment planning and patient counselling for 
    patients with non-muscle invasive bladder cancer:
    * Probability of avoiding tumour progression at 1 and 5 years from initial diagnosis
    * Individualized progression-free survival curve
    * Downloadable risk profile for use in longitudinal evaluation (coming soon)
    NIMBLE-2 is currently under development.
    """
    )
    st.subheader("Team")
    st.markdown("""
                **Students**: Nicole Bodnariuc, Shamir Malik, Krish Narayana, Jeremy Wu \n
                **Supervisors**: Dr. Girish Kulkarni, Dr. Jethro Kwong
                """)

PAGES = {
    "NIMBLE-1": page_nimble1,
    "NIMBLE-2": page_nimble2,
    "About NIMBLE": page_about,
}

if __name__ == "__main__":
    st.set_page_config(page_title="NIMBLE - Non-muscle invasive bladder cancer longitudinal evaluation",
                       page_icon="https://bladdercancercanada.org/wp-content/uploads/2017/03/bcc-fav-icon.png",
                       layout="wide",
                       initial_sidebar_state="auto"
                       )
    #load_widget_state()
    main()
