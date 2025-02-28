import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import os
import warnings
warnings.filterwarnings('ignore')

class PatientRecordAnalyzer:
    """
    A class to analyze patient records and provide insights for medication recommendations
    """
    
    def __init__(self, data_path=None):
        """Initialize the analyzer with data"""
        self.data = None
        if data_path:
            self.load_data(data_path)
        
        # Define medication contraindications dictionary
        # This maps conditions to medications that should be avoided
        self.contraindications = {
            'Diabetes': ['High-dose corticosteroids', 'Certain beta-blockers', 'Thiazide diuretics'],
            'Hypertension': ['NSAIDs', 'Decongestants', 'Stimulants'],
            'Kidney Disease': ['NSAIDs', 'Certain antibiotics', 'High-dose vitamin D'],
            'Liver Disease': ['Acetaminophen (high dose)', 'Statins', 'Certain antibiotics'],
            'Asthma': ['Non-selective beta-blockers', 'Aspirin', 'NSAIDs'],
            'Heart Failure': ['NSAIDs', 'Calcium channel blockers', 'Thiazolidinediones'],
            'Pregnancy': ['ACE inhibitors', 'Statins', 'Tetracyclines', 'Warfarin'],
            'Elderly': ['Benzodiazepines', 'Anticholinergics', 'Certain sedatives']
        }
        
        # Define positive medication recommendations
        self.recommendations = {
            'Diabetes': ['Metformin', 'GLP-1 receptor agonists', 'SGLT2 inhibitors'],
            'Hypertension': ['ACE inhibitors', 'ARBs', 'Calcium channel blockers'],
            'Kidney Disease': ['ACE inhibitors', 'ARBs', 'Erythropoietin'],
            'Liver Disease': ['Ursodeoxycholic acid', 'Low-dose acetaminophen', 'Vitamin E'],
            'Asthma': ['Inhaled corticosteroids', 'Long-acting beta agonists', 'Leukotriene modifiers'],
            'Heart Failure': ['ACE inhibitors', 'Beta-blockers', 'Diuretics', 'SGLT2 inhibitors'],
            'Elderly': ['Memantine', 'Donepezil', 'SSRIs']
        }
    
    def load_data(self, data_path):
        """Load data from CSV file"""
        try:
            self.data = pd.read_csv(data_path)
            print(f"Loaded data with {self.data.shape[0]} records and {self.data.shape[1]} attributes.")
            return True
        except Exception as e:
            print(f"Error loading data: {e}")
            return False
    
    def create_sample_data(self, n_samples=100):
        """Create sample data for demonstration purposes"""
        np.random.seed(42)
        
        # Define possible values for categorical variables
        genders = ['Male', 'Female']
        conditions = ['Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 
                      'Kidney Disease', 'Liver Disease', 'Cancer', 'Arthritis']
        procedures = ['Surgery', 'Medication Adjustment', 'Diagnostic', 'Therapy', 
                      'Consultation', 'Emergency Treatment']
        outcomes = ['Improved', 'Stable', 'Declined', 'Recovered']
        
        # Generate random data
        data = {
            'Age': np.random.randint(18, 90, n_samples),
            'Gender': np.random.choice(genders, n_samples),
            'Condition': np.random.choice(conditions, n_samples),
            'Procedure': np.random.choice(procedures, n_samples),
            'Cost': np.random.uniform(500, 10000, n_samples).round(2),
            'Length_of_Stay': np.random.randint(1, 30, n_samples),
            'Readmission': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
            'Outcome': np.random.choice(outcomes, n_samples),
            'Satisfaction': np.random.randint(1, 11, n_samples)
        }
        
        self.data = pd.DataFrame(data)
        print(f"Created sample dataset with {n_samples} records.")
        return self.data
    
    def data_summary(self):
        """Provide a summary of the dataset"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        summary = {
            "record_count": len(self.data),
            "attributes": list(self.data.columns),
            "missing_values": self.data.isnull().sum().to_dict(),
            "condition_counts": self.data['Condition'].value_counts().to_dict() if 'Condition' in self.data.columns else {},
            "age_stats": {
                "min": self.data['Age'].min() if 'Age' in self.data.columns else None,
                "max": self.data['Age'].max() if 'Age' in self.data.columns else None,
                "mean": self.data['Age'].mean() if 'Age' in self.data.columns else None
            },
            "gender_distribution": self.data['Gender'].value_counts().to_dict() if 'Gender' in self.data.columns else {},
            "average_los": self.data['Length_of_Stay'].mean() if 'Length_of_Stay' in self.data.columns else None,
            "readmission_rate": self.data['Readmission'].mean() if 'Readmission' in self.data.columns else None,
            "avg_satisfaction": self.data['Satisfaction'].mean() if 'Satisfaction' in self.data.columns else None
        }
        
        return summary
    
    def visualize_data(self, output_folder='patient_analysis_output'):
        """Generate visualizations from the data"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # 1. Age distribution
        plt.figure(figsize=(10, 6))
        sns.histplot(self.data['Age'], kde=True)
        plt.title('Age Distribution of Patients')
        plt.savefig(f"{output_folder}/age_distribution.png")
        plt.close()
        
        # 2. Condition distribution
        plt.figure(figsize=(12, 6))
        condition_counts = self.data['Condition'].value_counts()
        sns.barplot(x=condition_counts.index, y=condition_counts.values)
        plt.title('Distribution of Medical Conditions')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_folder}/condition_distribution.png")
        plt.close()
        
        # 3. Length of stay vs. Age
        plt.figure(figsize=(10, 6))
        sns.scatterplot(x='Age', y='Length_of_Stay', hue='Condition', data=self.data)
        plt.title('Length of Stay vs. Age by Condition')
        plt.savefig(f"{output_folder}/los_vs_age.png")
        plt.close()
        
        # 4. Outcome distribution by condition
        plt.figure(figsize=(12, 8))
        outcome_by_condition = pd.crosstab(self.data['Condition'], self.data['Outcome'])
        outcome_by_condition.plot(kind='bar', stacked=True)
        plt.title('Outcomes by Medical Condition')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_folder}/outcome_by_condition.png")
        plt.close()
        
        # 5. Satisfaction distribution
        plt.figure(figsize=(10, 6))
        sns.countplot(x='Satisfaction', data=self.data)
        plt.title('Distribution of Patient Satisfaction Scores')
        plt.savefig(f"{output_folder}/satisfaction_distribution.png")
        plt.close()
        
        # 6. Readmission rate by condition
        plt.figure(figsize=(12, 6))
        readmission_by_condition = self.data.groupby('Condition')['Readmission'].mean() * 100
        sns.barplot(x=readmission_by_condition.index, y=readmission_by_condition.values)
        plt.title('Readmission Rate (%) by Condition')
        plt.xticks(rotation=45)
        plt.ylabel('Readmission Rate (%)')
        plt.tight_layout()
        plt.savefig(f"{output_folder}/readmission_by_condition.png")
        plt.close()
        
        print(f"Visualizations saved to {output_folder} folder.")
    
    def analyze_patterns(self):
        """Identify patterns and clusters in the data"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return None
        
        # Select numerical columns for clustering
        num_cols = ['Age', 'Cost', 'Length_of_Stay', 'Satisfaction']
        num_cols = [col for col in num_cols if col in self.data.columns]
        
        if len(num_cols) < 2:
            print("Not enough numerical columns for clustering analysis.")
            return None
        
        # Prepare data for clustering
        X = self.data[num_cols].copy()
        X = X.fillna(X.mean())  # Fill missing values
        
        # Standardize the data
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply K-means clustering
        clusters = 4  # We can adjust this with elbow method if needed
        kmeans = KMeans(n_clusters=clusters, random_state=42)
        self.data['Cluster'] = kmeans.fit_predict(X_scaled)
        
        # Apply PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)
        
        # Create a dataframe with PCA results
        pca_df = pd.DataFrame({'PCA1': X_pca[:, 0], 'PCA2': X_pca[:, 1], 'Cluster': self.data['Cluster']})
        
        # Visualization of clusters
        plt.figure(figsize=(10, 8))
        sns.scatterplot(x='PCA1', y='PCA2', hue='Cluster', data=pca_df, palette='viridis')
        plt.title('Patient Clusters based on Age, Cost, Length of Stay, and Satisfaction')
        plt.savefig("patient_analysis_output/patient_clusters.png")
        plt.close()
        
        # Analyze clusters
        cluster_analysis = self.data.groupby('Cluster').agg({
            'Age': 'mean',
            'Cost': 'mean',
            'Length_of_Stay': 'mean',
            'Readmission': 'mean',
            'Satisfaction': 'mean',
            'Condition': lambda x: x.value_counts().index[0] if len(x.value_counts()) > 0 else None
        }).round(2)
        
        return cluster_analysis
    
    def medication_recommendations(self, patient_condition, patient_age=None, readmission_history=False):
        """
        Provide medication recommendations based on patient condition and avoid contraindications
        
        Args:
            patient_condition: Medical condition of the patient
            patient_age: Age of the patient (optional)
            readmission_history: Whether patient has been readmitted before (optional)
            
        Returns:
            Dictionary with recommended and contraindicated medications
        """
        recommendations = {
            'recommended': [],
            'contraindicated': [],
            'reasoning': []
        }
        
        # Get standard recommendations and contraindications for the condition
        if patient_condition in self.recommendations:
            recommendations['recommended'] = self.recommendations[patient_condition]
        
        if patient_condition in self.contraindications:
            recommendations['contraindicated'] = self.contraindications[patient_condition]
        
        # Add age-specific considerations
        if patient_age is not None:
            if patient_age >= 65:
                recommendations['contraindicated'].extend(self.contraindications.get('Elderly', []))
                recommendations['reasoning'].append("Due to advanced age, certain medications have been contraindicated.")
                
                # For elderly patients with readmission history, be more conservative
                if readmission_history and patient_condition == 'Diabetes':
                    recommendations['reasoning'].append("Due to readmission history and diabetes, sulfonylureas should be used with caution.")
                    
            elif patient_age < 18:
                recommendations['reasoning'].append("Pediatric dosing and medication selection should be considered.")
        
        # Add condition-specific reasoning
        if patient_condition == 'Kidney Disease':
            recommendations['reasoning'].append("Medication dosages may need adjustment due to altered renal clearance.")
        
        elif patient_condition == 'Liver Disease':
            recommendations['reasoning'].append("Medications metabolized by the liver should be used with caution or avoided.")
        
        # Remove duplicates
        recommendations['recommended'] = list(set(recommendations['recommended']))
        recommendations['contraindicated'] = list(set(recommendations['contraindicated']))
        
        return recommendations
    
    def generate_report(self, output_file='patient_analysis_report.txt'):
        """Generate a comprehensive report with insights"""
        if self.data is None:
            print("No data loaded. Please load data first.")
            return
        
        # Get summary statistics
        summary = self.data_summary()
        
        # Perform cluster analysis
        clusters = self.analyze_patterns()
        
        # Calculate additional insights
        condition_los = self.data.groupby('Condition')['Length_of_Stay'].mean().to_dict()
        condition_readmission = self.data.groupby('Condition')['Readmission'].mean().to_dict()
        condition_satisfaction = self.data.groupby('Condition')['Satisfaction'].mean().to_dict()
        
        # Calculate correlation matrix for numerical columns
        num_cols = ['Age', 'Cost', 'Length_of_Stay', 'Satisfaction']
        num_cols = [col for col in num_cols if col in self.data.columns]
        corr_matrix = self.data[num_cols].corr().round(2).to_dict()
        
        # Generate medication insights for top conditions
        med_insights = {}
        for condition in list(summary['condition_counts'].keys())[:5]:
            med_insights[condition] = self.medication_recommendations(
                condition,
                patient_age=summary['age_stats']['mean'],
                readmission_history=(condition_readmission.get(condition, 0) > 0.1)
            )
        
        # Write report to file
        with open(output_file, 'w') as f:
            f.write("=== Patient Record Analysis Report ===\n\n")
            
            f.write("1. Dataset Overview\n")
            f.write(f"   Total Records: {summary['record_count']}\n")
            f.write(f"   Age Range: {summary['age_stats']['min']} to {summary['age_stats']['max']} years (avg: {summary['age_stats']['mean']:.1f})\n")
            f.write(f"   Gender Distribution: {', '.join([f'{k}: {v}' for k, v in summary['gender_distribution'].items()])}\n")
            f.write(f"   Average Length of Stay: {summary['average_los']:.1f} days\n")
            f.write(f"   Readmission Rate: {summary['readmission_rate']*100:.1f}%\n")
            f.write(f"   Average Satisfaction Score: {summary['avg_satisfaction']:.1f}/10\n\n")
            
            f.write("2. Medical Conditions Analysis\n")
            for condition, count in summary['condition_counts'].items():
                f.write(f"   {condition}: {count} patients\n")
                f.write(f"      Average Length of Stay: {condition_los.get(condition, 0):.1f} days\n")
                f.write(f"      Readmission Rate: {condition_readmission.get(condition, 0)*100:.1f}%\n")
                f.write(f"      Average Satisfaction: {condition_satisfaction.get(condition, 0):.1f}/10\n")
            f.write("\n")
            
            f.write("3. Patient Clusters\n")
            for cluster, row in clusters.iterrows():
                f.write(f"   Cluster {cluster}:\n")
                f.write(f"      Predominant Condition: {row['Condition']}\n")
                f.write(f"      Average Age: {row['Age']:.1f} years\n")
                f.write(f"      Average Cost: ${row['Cost']:.2f}\n")
                f.write(f"      Average Length of Stay: {row['Length_of_Stay']:.1f} days\n")
                f.write(f"      Readmission Rate: {row['Readmission']*100:.1f}%\n")
                f.write(f"      Average Satisfaction: {row['Satisfaction']:.1f}/10\n")
            f.write("\n")
            
            f.write("4. Medication Recommendations\n")
            for condition, insights in med_insights.items():
                f.write(f"   {condition}:\n")
                f.write(f"      Recommended Medications: {', '.join(insights['recommended'])}\n")
                f.write(f"      Contraindicated Medications: {', '.join(insights['contraindicated'])}\n")
                if insights['reasoning']:
                    f.write(f"      Clinical Reasoning:\n")
                    for reason in insights['reasoning']:
                        f.write(f"         - {reason}\n")
            f.write("\n")
            
            f.write("5. Key Insights for Medication Management\n")
            f.write("   - Higher readmission rates for certain conditions suggest need for medication review\n")
            f.write("   - Elderly patients (clusters with higher average age) require special medication considerations\n")
            f.write("   - Length of stay correlates with condition complexity and may indicate medication effectiveness\n")
            f.write("   - Patient satisfaction scores may reflect medication side effects and should be monitored\n")
            f.write("   - Medication recommendations should be tailored to individual patient profiles\n")
        
        print(f"Report generated and saved to {output_file}")

# Example usage
if __name__ == "__main__":
    analyzer = PatientRecordAnalyzer()
    
    # Use create_sample_data if no real data is available
    analyzer.create_sample_data(200)
    
    # To use with real data:
    # analyzer.load_data("path_to_your_data.csv")
    
    # Generate visualizations
    analyzer.visualize_data()
    
    # Generate comprehensive report
    analyzer.generate_report()
    
    # Example of medication recommendation for a specific patient
    recommendations = analyzer.medication_recommendations("Diabetes", patient_age=75, readmission_history=True)
    print("\nExample Medication Recommendations for an Elderly Diabetic Patient:")
    print(f"Recommended: {', '.join(recommendations['recommended'])}")
    print(f"Contraindicated: {', '.join(recommendations['contraindicated'])}")
    for reason in recommendations['reasoning']:
        print(f"- {reason}")

        
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pyttsx3  # For Text-to-Speech
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

class PatientRecordAnalyzer:
    def __init__(self, file_path=None):
        self.file_path = file_path
        self.data = self.load_data()
        self.tts_engine = pyttsx3.init()

    def load_data(self):
        if self.file_path:
            return pd.read_csv(self.file_path)
        else:
            return self.generate_sample_data()
    
    def generate_sample_data(self):
        np.random.seed(42)
        conditions = ['Diabetes', 'Hypertension', 'Asthma', 'Heart Disease', 'Cancer']
        data = {
            'Patient ID': np.arange(1, 101),
            'Age': np.random.randint(20, 90, 100),
            'Condition': np.random.choice(conditions, 100),
            'Cost': np.random.randint(5000, 50000, 100),
            'Length of Stay': np.random.randint(1, 15, 100),
            'Readmission': np.random.choice([0, 1], 100, p=[0.8, 0.2]),
            'Satisfaction Score': np.random.randint(1, 10, 100),
            'Gender': np.random.choice(['Male', 'Female'], 100)
        }
        return pd.DataFrame(data)
    
    def speak(self, text):
        print("Speaking:", text)
        self.tts_engine.say(text)
        self.tts_engine.runAndWait()

    def data_summary(self):
        summary = {
            "Total Patients": len(self.data),
            "Common Conditions": self.data['Condition'].value_counts().to_dict(),
            "Gender Distribution": self.data['Gender'].value_counts().to_dict(),
            "Average Length of Stay": round(self.data['Length of Stay'].mean(), 2),
            "Readmission Rate": round(self.data['Readmission'].mean() * 100, 2),
            "Average Satisfaction": round(self.data['Satisfaction Score'].mean(), 2)
        }
        
        # Speak summary
        self.speak(f"Total Patients: {summary['Total Patients']}")
        self.speak(f"Most Common Condition: {max(summary['Common Conditions'], key=summary['Common Conditions'].get)}")
        self.speak(f"Readmission Rate: {summary['Readmission Rate']} percent")
        
        return summary
    
    def visualize_data(self):
        sns.histplot(self.data['Age'], bins=10, kde=True)
        plt.title("Age Distribution")
        plt.savefig("age_distribution.png")
        plt.show()
    
    def analyze_patterns(self, clusters=3):
        features = self.data[['Age', 'Cost', 'Length of Stay', 'Satisfaction Score']]
        kmeans = KMeans(n_clusters=clusters, random_state=42)
        self.data['Cluster'] = kmeans.fit_predict(features)
        
        pca = PCA(n_components=2)
        reduced_data = pca.fit_transform(features)
        
        plt.scatter(reduced_data[:, 0], reduced_data[:, 1], c=self.data['Cluster'], cmap='viridis')
        plt.title("Patient Clustering")
        plt.xlabel("PCA Component 1")
        plt.ylabel("PCA Component 2")
        plt.savefig("clustering_result.png")
        plt.show()
        
        self.speak("Clustering completed. Patients grouped into different categories.")
        
    def medication_recommendations(self, condition, patient_age, readmission_history):
        medications = {
            "Diabetes": {"Recommended": ["Metformin", "Insulin"], "Avoid": ["Steroids"]},
            "Hypertension": {"Recommended": ["ACE Inhibitors", "Beta Blockers"], "Avoid": ["NSAIDs"]},
            "Asthma": {"Recommended": ["Inhalers", "Montelukast"], "Avoid": ["Beta Blockers"]},
            "Heart Disease": {"Recommended": ["Statins", "Aspirin"], "Avoid": ["NSAIDs", "Steroids"]},
            "Cancer": {"Recommended": ["Chemotherapy", "Immunotherapy"], "Avoid": ["Steroids"]},
        }
        
        if condition in medications:
            recommended = medications[condition]["Recommended"]
            avoid = medications[condition]["Avoid"]
            
            self.speak(f"For {condition}, recommended medications are {', '.join(recommended)}.")
            self.speak(f"Avoid {', '.join(avoid)}.")
            
            return {"Recommended": recommended, "Avoid": avoid}
        else:
            self.speak("No data available for this condition.")
            return {"Recommended": [], "Avoid": []}
    
    def generate_report(self):
        summary = self.data_summary()
        report = """
        Patient Data Analysis Report
        --------------------------------------
        Total Patients: {0}
        Most Common Condition: {1}
        Readmission Rate: {2}%
        Average Satisfaction Score: {3}
        
        Medication Recommendations:
        
        """.format(summary['Total Patients'], max(summary['Common Conditions'], key=summary['Common Conditions'].get), summary['Readmission Rate'], summary['Average Satisfaction'])
        
        with open("patient_report.txt", "w") as file:
            file.write(report)
        
        self.speak("Report generated successfully.")

# Example Usage
analyzer = PatientRecordAnalyzer()
summary = analyzer.data_summary()
analyzer.visualize_data()
analyzer.analyze_patterns()
medications = analyzer.medication_recommendations("Diabetes", patient_age=65, readmission_history=True)
analyzer.generate_report()
