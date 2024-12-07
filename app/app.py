import numpy as np
import os
import csv
from datetime import datetime
from flask import Flask, render_template, request,jsonify
import traceback
from langchain.schema import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
import pandas as pd
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, CrossEncoder
from huggingface_hub import login

app= Flask(__name__)

# Add your Hugging Face token
HUGGINGFACE_TOKEN='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'
login(token= HUGGINGFACE_TOKEN)

# load environment variables and models
os.environ['GOOGLE_API_KEY']='AIzaSyCwDDJV9mke7aFcoiMSOpIMk4SD9VrbbNc'
# initializing gemini model
chat_gemini = ChatGoogleGenerativeAI(model='gemini-1.5-flash',temperature=0.4)

# cross encoder model
cross_encoder= CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

# Define the new system prompt

SYSTEM_PROMPT = '''
You are a document analyzer specialising in policy documents.
Your task is to identify the policy type based on its content.
The possible policies are:

Medical Policy: The ITC Medical Policy ensures the health and well-being of its managers through medical benefits and regular checkups. 
It covers managers, their spouses, and two dependent children under 24, offering tax-exempt reimbursements, maternity benefits (₹1.25 lakh), 
dental treatments, Ayurveda, and homeopathy. The policy includes specific vaccination schedules and hospitalization support, with prior 
approval needed for planned expenses. Non-reimbursable items include cosmetics, luxury expenses, and fertility treatments. 
Managers must undergo periodic health checkups based on age brackets. 
xExcess medical expenses may be considered for reimbursement at the Medical Committee's discretion. Overseas medical insurance is provided for official travel purposes.

Keywords this policy includes: health benefits, medical reimbursement, medical budget, hospitalization, illness, treatment, wellness, maternity, dental, vaccination.

Travel Expenses Policy: The ITC Travel Expenses Policy provides guidelines for official travel, covering transit accommodations, reimbursements, and daily allowances. Eligibility applies to Managers (Level 7 and above) and Office Associates, with travel class and allowances based on grades. Reimbursements include fixed and actual expenses for accommodations, food, and incidentals, with specific provisions for ITC and non-ITC hotels. Overseas travel allowances are specified in US dollars by grade. Claims require proper documentation, including tickets, invoices, and payment proofs. Travel bookings must be through recognized agents, prioritizing ITC accommodations. Misuse or false claims result in disciplinary action, and compliance with administrative guidelines is mandatory.

Keywords this policy includes: travel reimbursement, transit accommodation, daily allowance, official trips, transportation, class of travel, boarding pass, hotel stay, incidental expenses, fixed TE, non-ITC hotels.

Product Sampling Policy: The ITC Product Sampling Policy enables employees and their families to use and provide feedback on ITC products/services. Eligible employees can claim annual reimbursements for ITC products purchased in India, subject to grade-specific limits and applicable taxes. Claims require proper invoices and must meet minimum billing thresholds. Coverage includes various ITC divisions such as food, personal care, and hotels, excluding tobacco, consumer durables, and non-ITC products. Promotions or employment changes adjust eligibility. Misuse or false claims can result in disciplinary action. Unused limits cannot be carried forward, and claims must be submitted by April 5th of the following financial year.

Keywords this policy includes: excess claims, future entitlements, product usage, sampling arrangement, feedback, ITC products, reimbursement for products, eligible employees, gifting products, purchase invoices, product categories.

If the above policies are not matched, give "No such policy found".

When presented with a question, accurately classify it as policy name to which it belongs.
NOTE: only give policy name. No description is required, no enter or any other expression with the policy name
'''


# load PDF content into memory

def load_pdfs(directory):
    pdf_content = {}
    for filename in os.listdir(directory):
        if filename.endswith('.pdf'):
            reader = PdfReader(os.path.join(directory, filename))
            text = ''.join([page.extract_text() for page in reader.pages])
            pdf_content[os.path.splitext(filename)[0]] = text
    return pdf_content
    
pdf_directory = os.path.join(os.path.dirname(__file__), 'data', 'pdf')
pdf_content=load_pdfs(pdf_directory)

# Initialize vector_store as a global variable
vector_store = None

# Loading the ChromaDB 
def load_chroma_db():
    global vector_store
    try:
        print("Loading ChromaDB...")
        embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        
        # Get the absolute path to the vector store directory
        base_dir = os.path.dirname(os.path.abspath(__file__))
        persist_directory = os.path.join(base_dir, '..', 'itc_poc_db')
        
        print(f"Loading ChromaDB from: {persist_directory}")
        
        vector_store = Chroma(
            persist_directory=persist_directory,
            embedding_function=embedding,
            collection_name='itc_poc_policies'
        )
        
        # Verify the loaded database
        test_results = vector_store.similarity_search("test", k=1)
        print(f"ChromaDB loaded successfully with {len(test_results)} results")
        return True
    except Exception as e:
        print(f"Error loading ChromaDB: {str(e)}")
        traceback.print_exc()  # Add this for more detailed error logging
        return False


# Function to classify policy using NER and keywords

def classify_policy(query):
    prompt = f'''
    Identify whether the user query "{query}" is pertaining to a Medical Policy, Travel Expenses Policy, or Product Sampling Policy. Provide accurate identification of policy.

    The following are the descriptions and inclusions in the policy:
    Medical Policy:
    Description: The Medical Policy aims to ensure the well-being of ITC managers and their families by offering comprehensive medical benefits, regular health checkups, and reimbursement provisions.
    What are included:
    1. Eligibility,
    2. Budget,
    3. Coverage,
    4. Taxability,
    5. Periodic Medical Checkups,
    6. Reimbursement Guidelines,
    7. Travel expenses for medical purposes,
    8. Maternity and dental benefits,
    9. Hospitalization rules,
    10. Vaccination entitlements,
    11. Exclusions,
    12. Expenses abroad(medical related)

    Travel Expenses Policy:
    Description: The Travel Expenses Policy outlines the facilities, reimbursements, and daily allowances provided to ITC employees for official travel. It details eligibility, entitlements, travel classes, and reimbursement guidelines for domestic and international trips, aiming to ensure a standardized process for managing travel-related expenses.
    What are included:
    1. Eligibility Criteria,
    2. Classes of Travel (Domestic and Overseas),
    3. Travel by Road Guidelines,
    4. Travel Expense Rates (Fixed and Actuals),
    5. Overseas Travel Expense Rates,
    6. Course Training Allowances,
    7. Claim Submission Guidelines,
    8. Administrative Aspects of Travel Bookings and Reimbursements,
    9. Hotel Accommodation Rules


    Product Sampling Policy:
    Description: This policy is aimed at enabling employees and their families to utilize ITC's products and services, provide feedback, and optionally share these with their friends and associates. The policy outlines eligibility, guidelines for reimbursement, covered products and services, and governance.
    What are included:
    1. Eligibility Criteria
    2. Guidelines for Promotions
    3. Reimbursement Rules
    4. Covered Products & Services
    5. Specific Exclusions
    6. Invoice Submission Requirements
    7. Online Purchases & Declaration
    8. Billing Minimum Amount
    9. Submission Deadlines
    10. Unused Entitlements
    11. Designated Resource for Processing
    12. Monthly Reimbursement Processing
    13. Governance & Disciplinary Measures
    14. Policy Amendments and Discretion


    Please remember the above are the only things that are included in the policy. Anything else is not included.

    Remember: Provide only the policy name.
    '''
    
    message = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=prompt)
    ]
    
    try:
        response = chat_gemini.invoke(message)
        policy_name = response.content.strip()
        
        # Define valid policies
        valid_policies = [
            "Medical Policy",
            "Travel Expenses Policy",
            "Product Sampling Policy",
            "No such policy found"
        ]
        
        # Flexible matching
        for valid_policy in valid_policies:
            if valid_policy.lower() in policy_name.lower():
                return valid_policy
        
        # If no match found, return default
        return "No such policy found"
        
    except Exception as e:
        print(f"Error in policy classification: {e}")
        return "No such policy found"


# function to get the relevant content from RAG

def get_context(user_input):
    print(f"\nDebug: Starting context retrieval for query: {user_input}")
    
    try:
        if vector_store is None:
            print("Debug: vector_store is None!")
            return "", []  # Return empty context and chunks
            
        top_chunks = vector_store.similarity_search(query=user_input, k=10)
        print(f"Debug: Retrieved {len(top_chunks)} initial chunks")
        
        if not top_chunks:
            print("Debug: No chunks found")
            return "", []

        cross_encoding_scores = []
        print("Debug: Starting cross-encoding scoring")
        
        for idx, chunk in enumerate(top_chunks):
            pair = [user_input, chunk.page_content]
            score = cross_encoder.predict(pair)
            cross_encoding_scores.append(score)
            print(f"Debug: Chunk {idx} score: {score}")

        top_5_indices = np.argsort(cross_encoding_scores)[::-1][:5]
        top_5_chunks = [top_chunks[o] for o in top_5_indices]
        print(f"Debug: Selected top {len(top_5_chunks)} chunks")

        context_for_llm = '\n\n'.join(chunk.page_content for chunk in top_5_chunks)
        
        
        return top_5_chunks,context_for_llm

    except Exception as e:
        print(f"Debug: Error in get_context: {str(e)}")
        return "", []

# Function to get response from the LLM

def get_groq_response(system_message_content,user_input,policy_name,not_found_message):
    try:
        if 'No such policy found' not in policy_name:
            message=[
                SystemMessage(content=system_message_content),
                HumanMessage(content=user_input)
            ]
            return chat_gemini.invoke(message).content
        else:
            return chat_gemini.invoke(not_found_message)
    except Exception as e:
        print(f"Error invoking LLM: {e}")
        return "Sorry, I could not process your request at the moment."


# Function to combine two excel sheets into one dataframe

def combine_excel_sheets(excel_file, sheet_name1, sheet_name2):
    excel_path = os.path.join(os.path.dirname(__file__), 'data', 'excel', excel_file)
    df_sheet1 = pd.read_excel(excel_path, sheet_name=sheet_name1, header=1)
    df_sheet2 = pd.read_excel(excel_path, sheet_name=sheet_name2, header=1)
    df_combined = pd.concat([df_sheet1, df_sheet2], ignore_index=True)
    return df_combined

df_product = combine_excel_sheets('Product Sampling.xlsx', 'OneITC', 'OneITC_Specialists')
df_medical = combine_excel_sheets('Medical Policy.xlsx', 'OneITC', 'OneITC_Specialists')


# Function to fetch user info based on employee number or other details
def get_user_info(employee_id=None, division=None, resp_level=None, grade=None, policy_name=None):
    # Convert employee_id to integer if it's not None
    if employee_id is not None:
        try:
            employee_id = int(employee_id)
        except ValueError:
            print(f"Warning: Could not convert employee_id {employee_id} to integer")
            return "No matching data found for this employee."
    
    if employee_id:
        filtered_medical = df_medical[df_medical['Emp No.'].notna()]
        filtered_medical = filtered_medical[filtered_medical['Emp No.'].astype(float).astype(int) == employee_id]
        filtered_product = df_product[df_product['Emp No.'].notna()]
        filtered_product = filtered_product[filtered_product['Emp No.'].astype(float).astype(int) == employee_id]
        if not filtered_medical.empty or filtered_product.empty:
            user_info = {
                'Medical Domiciliary (Rs.)': filtered_medical['Medical Domiciliary (Rs.)'].iloc[0],
                'Medical Hospitalisation (Rs.)': filtered_medical['Medical Hospitalisation (Rs.)'].iloc[0],
                'Hospital Room Entitlement': filtered_medical['Hospital Room Entitlement'].iloc[0],
                'Product Sampling (Rs.)': filtered_product['Product Sampling (Rs.)'].iloc[0],
                'One_ITC_Specialist': filtered_product['Specialist'].iloc[0]
            }
    else:
        filtered_medical = df_medical[(df_medical["Division"] == division) &
                                      (df_medical['Resp Level'] == resp_level) &
                                      (df_medical['Grade'] == grade)]
        filtered_product = df_product[(df_product["Division"] == division) &
                                      (df_product['Resp Level'] == resp_level) &
                                      (df_product['Grade'] == grade)]
        user_info = {
            'Medical Domiciliary (Rs.)': filtered_medical['Medical Domiciliary (Rs.)'].iloc[0],
            'Medical Hospitalisation (Rs.)': filtered_medical['Medical Hospitalisation (Rs.)'].iloc[0],
            'Hospital Room Entitlement': filtered_medical['Hospital Room Entitlement'].iloc[0],
            'Product Sampling (Rs.)': filtered_product['Product Sampling (Rs.)'].iloc[0],
            'One_ITC_Specialist': filtered_product['Specialist'].iloc[0]
        }

    return user_info
    
# Modify the log_to_csv function to handle both initial logging and feedback updates
def log_to_csv(employee_id, division, resp_level, grade, user_query, policy_name, response, feedback=None):
    log_file = 'user_query_log.csv'
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    # Create file with headers if it doesn't exist
    if not os.path.isfile(log_file):
        with open(log_file, mode='w', newline='', encoding='utf-8') as file:
            writer = csv.writer(file)
            writer.writerow(['Employee ID', 'Division', 'Responsibility Level', 'Grade', 
                           'Query', 'Policy Name', 'Response', 'Timestamp', 'Feedback'])
    
    # Read existing entries to find and update feedback if necessary
    if feedback is not None:
        rows = []
        updated = False
        with open(log_file, mode='r', newline='', encoding='utf-8') as file:
            reader = csv.reader(file)
            headers = next(reader)  # Skip headers
            for row in reader:
                if (row[0] == str(employee_id) and 
                    row[1] == division and 
                    row[2] == resp_level and 
                    row[3] == grade and 
                    row[4] == user_query and 
                    row[6] == response and 
                    row[8] == ''):  # Check if feedback is empty
                    row[8] = feedback  # Update feedback
                    updated = True
                rows.append(row)
        
        if updated:
            with open(log_file, mode='w', newline='', encoding='utf-8') as file:
                writer = csv.writer(file)
                writer.writerow(headers)
                writer.writerows(rows)
            return
    
    # If not updating feedback, write new row
    log_data = [
        employee_id,
        division,
        resp_level,
        grade,
        user_query,
        policy_name,
        response,
        timestamp,
        feedback or ''  # Empty string if feedback is None
    ]
    
    with open(log_file, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(log_data)



@app.route('/')
def home():
    # Fetch unique values for dropdowns
    unique_grades = sorted(pd.concat([df_medical['Grade'], df_product['Grade']]).dropna().unique())
    unique_resp_levels = sorted(pd.concat([df_medical['Resp Level'], df_product['Resp Level']]).dropna().unique())
    unique_divisions = sorted(pd.concat([df_medical['Division'], df_product['Division']]).dropna().unique())

    # Pass these values to the frontend
    return render_template(
        'index.html',
        grades=unique_grades,
        resp_levels=unique_resp_levels,
        divisions=unique_divisions
    )

@app.route('/api/query',methods=['POST'])
def query():
    try:
        data=request.get_json()
        user_query=data['query']
        employee_id = data.get('employee_id', None)
        division = data.get('division', None)
        resp_level = data.get('resp_level', None)
        grade = data.get('grade', None)

        # Classify the policy
        policy_name= classify_policy(user_query)

        #Get the specific policy text
        policy_text= pdf_content.get(policy_name,'No content found for this policy.')

        # Fetch user_specific info
        user_info= get_user_info(employee_id,division,resp_level,grade,policy_name=policy_name)

        # Prepare system message for LLM
        system_message = """
        You are an Human Resource executive of ITC, who understands the HR and other organizational policies and have in-depth understanding of the usage of these policies based on user queries.
        You will answer to the user queries in detail by understanding the appropriate policy the user query belong to. The answers are
        going to be factual. In some cases answers might be summarised, if necessary. You will maintain a empathetic tone in your response.
        Currently you are going to refrain from providing decisions to the employees.

        You are going to serve the following category employee with the varying grade and responsibility levels -
        1. One ITC employee
        2. One ITC employee specialists

        The kind of queries you will respond to
        1. Queries about how to avail a specific policy benefits.
        2. Understand the current status of the availed benefits from the policy.
        3. Queries related to specific clause of a policy.
        4. Interpretation of policy benefits which are convoluted. # chain of thought- few shot(First visit is to Vizag and then to Mumbai by the same employee)
        5. Information regarding the procedure for claims.
        6. Final settlement and information Point of Contact for the same.
        7. Hierarchy of approvals to avail specific set of policy benefits.
        8. Eligibility to avail a benefit of a policy.
        9. Different types of the document the employee would have to produce to claim the particular policy benefits

        Points to remember while responding:
          1. Use simple, clear language and focus on accuracy in HR policy interpretation and compliance requirements.
          2. Be smart and Provide practical recommendations for policy management, compliance, and employee support.
          3. Provide context-based reasoning for policy interpretations and recommendations.
          4. Consider ITC’s core HR values and employee well-being in your responses and ensure clarity in guidance for HR-related communications and documentation.
          5. Differentiate between OneITC and OneITC_Specialists
          6. Please consider the financial-year as a unit while generating responses to the queries.
          7. Consider the query intent and policy intent before you generate a response.
          8. Based on the query intent and the policy intent Frame the response in such a way that employees are clear from their understanding of the usage of the policy.
          9. If their are multiple policies then refrain from generalizing answers and direct the employees to Human HR.

        """
        # get the relevant chunks
        top_5_chunks,context_for_llm= get_context(user_input=user_query)

        # Extract just the text content from the Document objects
        chunk_texts = [chunk.page_content for chunk in top_5_chunks]

        # human prompt for generating response
        prompt = f'''
        You are an AI assistant for an HR Manager at ITC, helping with policy interpretation, compliance management,
        employee well-being, and HR strategy. Follow these guidelines strictly:

        1. For inclusion/exclusion questions:
           - If asked "Does policy X include/exclude Y?", respond ONLY with:
             * For inclusion: "Yes, {policy_name} includes [specific item]" OR "No, {policy_name} does not include [specific item]"
             * For exclusion: "Yes, {policy_name} excludes [specific item]" OR "No, {policy_name} does not exclude [specific item]"
           - Do not provide additional context unless specifically requested

        2. For general questions, follow this structured analysis:
           a) Intent Identification ({policy_name}):
              - Main issue/question
              - Query category (leave/compliance/engagement)

           b) Policy Context Review:
              - ITC's HR values
              - Employee well-being priorities
              - Department considerations

           c) Policy Analysis:
              - Specific criteria
              - Eligibility conditions
              - Compliance requirements

           d) Recommendations:
              - Clear, actionable advice
              - Employee well-being considerations
              - Required follow-ups

           e) Concise Summary

        For every question, you should think in the above way but don't give the answers in the above format.

        User's Question: "{user_query}"
        User's grade: "{grade}"
        User's responsibility level: "{resp_level}"
        User's division: "{division}"
        User's info: {user_info}

        Reference Data: {context_for_llm}

        While giving numbers like Medical Domiciliary (Rs.),Medical Hospitalisation (Rs.),Hospital Room Entitlement, Product Sampling (Rs.)
        and other personalized numbers, please use the user_info to generate the response, not the numbers that you get from the policy itself.
        That is if anything which is mentioned in the policy document, is also mentioned in the user_info, then use the user_info to generate the response.
        
        While giving answers, please use simple english, short format answer in proper bulleting format and give added perks where it is absolutely necessary.
        If there is some numbers associated in the response, try to put that figure in the response at the very top and then followed by the rest.
        Start the response with a conclusive statement (don't start with Based on the information)and then followed by the rest of the response.
        Example Response Format:
        Question: "Does the leave policy include work from home days?"
        Answer: "No, the leave policy does not include work from home days."

        Question: "How many leaves remaining?"
        Answer: Based on {user_info}:
        - Annual Entitlement: 26 leaves
        - Carry Forward Limit: 7 leaves
        - Remaining Available: [26 - 7 - N] leaves
        '''
        not_found_message = "No such policy found"

        #getting the response from the llm
        response= get_groq_response(system_message,prompt,policy_name,not_found_message)

        #checking the intermediate text

        print(response)
        print(user_info)
        print(grade)
        print(resp_level)
        print(division)
        print(policy_name)
        print(len(top_5_chunks))
        print(context_for_llm)

        # log in the interaction with empty feedback
        log_to_csv(employee_id,division,resp_level,grade,user_query,policy_name,response)

        return jsonify({
            'response':response,
            'policy':policy_name,
            'user_info':user_info,
            'policy_text':policy_text,
            'chunks': chunk_texts
        })

    except Exception as e:
        print(f"Detailed error: {str(e)}")
        print(f"Error type: {type(e)}")
        print(f"Traceback: {traceback.format_exc()}")
        return jsonify({
            "error": str(e),
            "type": str(type(e).__name__)
        }), 500


@app.route('/api/feedback', methods=['POST'])
def feedback():
    try:
        data = request.get_json()
        # Update existing row with feedback
        log_to_csv(
            employee_id=data.get('employee_id'),
            division=data.get('division'),
            resp_level=data.get('resp_level'),
            grade=data.get('grade'),
            user_query=data.get('query'),
            policy_name=data.get('policy'),
            response=data.get('response'),
            feedback=data.get('feedback')
        )
        return jsonify({"status": "success"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/highlight', methods=['POST'])
def highlight():
    try:
        data = request.json
        policy_text = data.get('policy_text', '')
        chunks = data.get('chunks', [])
        
        highlighted_text = policy_text
        
        # Sort chunks by length in descending order to avoid nested highlights
        chunks.sort(key=len, reverse=True)
        
        # Highlight each chunk in the policy text
        for chunk in chunks:
            if chunk in highlighted_text:
                highlighted_text = highlighted_text.replace(
                    chunk,
                    f'<mark>{chunk}</mark>'
                )
        
        return jsonify({
            "highlighted_text": highlighted_text
        })
    except Exception as e:
        print(f"Error in highlight route: {str(e)}")
        return jsonify({
            "error": str(e)
        }), 500

if __name__ == '__main__':
    if not load_chroma_db():
        print("Failed to load ChromaDB. Exiting...")
        exit(1)
    
    # Additional verification
    try:
        test_query = "medical policy"  # Use a more relevant test query
        test_results = vector_store.similarity_search(test_query, k=1)
        print(f"Vector store test successful. Found {len(test_results)} results")
        if test_results:
            print("Sample result:", test_results[0].page_content[:100], "...")  # Print first 100 chars of result
    except Exception as e:
        print(f"Vector store test failed: {str(e)}")
    
    app.run(debug=True)
