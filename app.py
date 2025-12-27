import streamlit as st
import pandas as pd
import numpy as np
import joblib
st.title("Bangalore House Price Predictor")
# Load the trained model and columns
@st.cache_resource
def load_model_and_features():
    try:
        # Load the model
        model = joblib.load('linear_regression_model.pkl')
        
        # Load ALL feature columns that model expects
        model_columns = joblib.load('model_features.pkl')
        
        # Load location names (for dropdown)
        location_columns = joblib.load('available_locations.pkl')
        
        return model, model_columns, location_columns
    except Exception as e:
        st.error(f"Error loading model files: {e}")
        st.info("Please run the Jupyter notebook first to save the model files.")
        return None, None, None

# Function to convert square feet input
def convert_sqft_to_num(x):
    if not x:
        return None
    if '-' in str(x):
        tokens = str(x).split('-')
        if len(tokens) == 2:
            try:
                return (float(tokens[0]) + float(tokens[1])) / 2
            except:
                return None
    try:
        return float(x)
    except:
        return None

def main():
    st.title("Bangalore House Price Predictor")
    st.markdown("Predict house prices in Bangalore based on property features")
    
    # Load model and data
    model, model_columns, location_columns = load_model_and_features()
    
    if model is None:
        st.warning("Model not loaded. Please check that model files exist.")
        return
    
    # Sidebar for user inputs
    with st.sidebar:
        st.header("📋 Enter Property Details")
        
        # Sort locations alphabetically for easy selection
        sorted_locations = sorted(location_columns)
        
        # User inputs
        location = st.selectbox("Select Location", sorted_locations)
        
        # For square feet - allow both single values and ranges
        sqft_input = st.text_input(
            "Total Square Feet",
            placeholder="e.g., 1500 or 1200-1800",
            help="Enter a number or range like 1200-1800"
        )
        
        bhk = st.selectbox("Number of BHK", [1, 2, 3, 4, 5, 6, 7, 8])
        
        bath = st.number_input(
            "Number of Bathrooms",
            min_value=1,
            max_value=bhk + 3,
            value=min(bhk, 3),
            help=f"Recommended: 1 to {bhk + 2} bathrooms for {bhk} BHK"
        )
        
        # Validate before prediction
        total_sqft = None
        if sqft_input:
            total_sqft = convert_sqft_to_num(sqft_input)
            if total_sqft is not None:
                # Show validations
                if total_sqft / bhk < 300:
                    st.warning(f"⚠️ Low area per BHK ({total_sqft/bhk:.0f} sq ft). Minimum recommended: 300 sq ft per BHK")
                
                if bath > bhk + 2:
                    st.warning(f"⚠️ High bathroom count. Typically {bhk} BHK has {bhk} to {bhk+2} bathrooms")
        
        # Predict button
        predict_btn = st.button("🚀 Predict Price", type="primary", use_container_width=True)
    
    # Prediction logic
    if predict_btn:
        if not sqft_input:
            st.error("Please enter square feet value")
            st.stop()
        
        total_sqft = convert_sqft_to_num(sqft_input)
        if total_sqft is None:
            st.error("Invalid square feet value. Please enter a valid number or range")
            st.stop()
        
        # Create a dataframe with ALL columns that model expects
        # Initialize all columns with 0
        input_dict = {}
        for col in model_columns:
            input_dict[col] = 0
        
        # Set the numerical features
        input_dict['total_sqft'] = total_sqft
        input_dict['bath'] = bath
        input_dict['bhk'] = bhk
        
        # Set the location feature (one-hot encoding)
        if location in input_dict:
            input_dict[location] = 1
        else:
            st.error(f"Location '{location}' not found in model features. Please select from the list.")
            st.stop()
        
        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])
        
        # Ensure correct column order
        input_df = input_df[model_columns]
        
        # Make prediction
        try:
            prediction = model.predict(input_df)[0]
            
            # Display results
            st.success("### 📊 Prediction Results")
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric(
                    "Predicted Price", 
                    f"₹{prediction:,.2f} Lakhs",
                    help="1 Lakh = ₹100,000"
                )
            with col2:
                price_per_sqft = (prediction * 100000) / total_sqft
                st.metric(
                    "Price per Sq Ft", 
                    f"₹{price_per_sqft:,.0f}"
                )
            with col3:
                st.metric(
                    "Total Area", 
                    f"{total_sqft:,.0f} sq ft"
                )
            
            # Property details
            st.info("#### 📝 Property Details")
            details_data = {
                "Feature": ["Location", "Total Square Feet", "BHK", "Bathrooms", "Area per BHK"],
                "Value": [location, f"{total_sqft:,.0f}", bhk, bath, f"{total_sqft/bhk:,.0f} sq ft"]
            }
            st.dataframe(pd.DataFrame(details_data), hide_index=True, use_container_width=True)
            
            # Price insights
            st.info("#### 💡 Price Insights")
            st.write(f"- **Price Range Estimate:** ₹{prediction*0.9:,.1f}L - ₹{prediction*1.1:,.1f}L")
            st.write(f"- **Total Value:** ₹{prediction*100000:,.0f}")
            
            if price_per_sqft > 10000:
                st.write("📍 **Premium Location** - Above average price per sq ft")
            elif price_per_sqft < 5000:
                st.write("📍 **Budget Location** - Below average price per sq ft")
            else:
                st.write("📍 **Average Location** - Market standard price per sq ft")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
            st.info("Check that input values are within reasonable ranges")
    
    # Information section
    st.markdown("---")
    st.header("📈 Model Information")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Performance")
        st.metric("R² Score", "84.52%", "Linear Regression")
        st.metric("Training Data", "7,251 properties")
        st.metric("Locations", f"{len(location_columns)} areas")
    
    with col2:
        st.subheader("Data Guidelines")
        st.write("✅ Based on cleaned data:")
        st.write("- Minimum 300 sq ft per BHK")
        st.write("- Bathrooms ≤ BHK + 2")
        st.write("- Outliers removed")
        st.write("- Price range: ₹20-400 Lakhs")
    
    # Instructions
    with st.expander("ℹ️ How to use"):
        st.markdown("""
        1. **Select Location** from dropdown (244 areas available)
        2. **Enter Square Feet** - single value or range
        3. **Choose BHK** - number of bedrooms
        4. **Set Bathrooms** - typically same as or slightly more than BHK
        5. **Click Predict** for price estimate
        
        **Model trained on:**
        - 7,251 Bangalore properties
        - 244 different locations
        - Linear Regression algorithm
        - 84.5% accuracy (R² score)
        """)
    
    # Footer
    st.markdown("---")
    st.caption("Note: This is a machine learning model prediction. Actual prices may vary based on market conditions and property specifics.")

if __name__ == "__main__":
    main()