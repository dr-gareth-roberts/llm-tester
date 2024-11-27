import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class ResultVisualizer:
    def __init__(self, results_df: pd.DataFrame):
        self.results_df = results_df

    def plot_performance_by_model(self, metric: str = 'response_length'):
        """
        Create a box plot comparing a metric across different models
        
        :param metric: Metric to visualize
        """
        plt.figure(figsize=(12, 6))
        sns.boxplot(x='model', y=metric, data=self.results_df)
        plt.title(f'{metric.replace("_", " ").title()} by Model')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def plot_parameter_impact(self, 
                               parameter: str = 'temperature', 
                               metric: str = 'response_length'):
        """
        Visualize the impact of a parameter on a specific metric
        
        :param parameter: Model parameter to analyze
        :param metric: Metric to compare
        """
        plt.figure(figsize=(12, 6))
        sns.scatterplot(
            x=parameter, 
            y=metric, 
            hue='model', 
            data=self.results_df
        )
        plt.title(f'Impact of {parameter} on {metric}')
        plt.tight_layout()
        plt.show()

    def generate_comprehensive_report(self):
        """
        Generate a multi-plot report of key metrics
        """
        fig, axs = plt.subplots(2, 2, figsize=(20, 15))
        
        # Metrics to visualize
        metrics = [
            'response_length', 
            'flesch_reading_ease', 
            'lexical_diversity', 
            'prompt_response_similarity'
        ]
        
        for i, metric in enumerate(metrics):
            row = i // 2
            col = i % 2
            
            sns.boxplot(
                x='model', 
                y=metric, 
                data=self.results_df, 
                ax=axs[row, col]
            )
            axs[row, col].set_title(f'{metric.replace("_", " ").title()} by Model')
            axs[row, col].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()

    def save_report(self, filename: str = 'llm_test_report.pdf'):
        """
        Save the visualization report as a PDF
        
        :param filename: Output filename
        """
        from matplotlib.backends.backend_pdf import PdfPages
        
        with PdfPages(filename) as pdf:
            # Generate comprehensive report
            self.generate_comprehensive_report()
            pdf.savefig()
            plt.close()