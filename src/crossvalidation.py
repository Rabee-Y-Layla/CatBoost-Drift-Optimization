import pandas as pd
import numpy as np
from catboost import CatBoostRegressor
from sklearn.model_selection import LeaveOneOut, KFold
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Display settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 12

print(" Starting Cross-Validation Feature Importance Analysis...")

# =============================================================================
# 1. Load Data
# =============================================================================

def load_data():
    """Load and prepare data"""
    try:
        df = pd.read_csv('Article1.csv')
        X = df.iloc[:, 1:-8]
        Y = df.iloc[:, 16]
        
        # Clean column names
        def clean_column_names(df):
            new_columns = []
            for col in df.columns:
                cleaned = str(col).replace('\t', '').replace('\\', '').replace('/', '_')
                cleaned = cleaned.replace('(', '').replace(')', '').replace('%', 'perc')
                cleaned = cleaned.replace(' ', '_').strip()
                if not cleaned:
                    cleaned = f'feature_{len(new_columns)}'
                new_columns.append(cleaned)
            return new_columns
        
        X.columns = clean_column_names(X)
        
        print(f" Data loaded: {X.shape}")
        print(f" Feature names: {list(X.columns)}")
        
        return X, Y
        
    except Exception as e:
        print(f" Error loading data: {e}")
        return None, None

# =============================================================================
# 2. Cross-Validation Feature Importance
# =============================================================================

def cross_validation_feature_importance(X, Y, cv_method='loocv', n_splits=None):
    """
    Calculate feature importance using cross-validation
    
    Parameters:
    - cv_method: 'loocv' for Leave-One-Out or 'kfold' for K-Fold
    - n_splits: number of splits for K-Fold (ignored for LOOCV)
    """
    
    print(f"\n  Starting {cv_method.upper()} Feature Importance Analysis...")
    
    # Select cross-validation method
    if cv_method == 'loocv':
        cv = LeaveOneOut()
        n_splits = len(X)
        print(f" Using Leave-One-Out CV with {n_splits} splits")
    else:
        if n_splits is None:
            n_splits = min(5, len(X))  # Default to 5-fold or less if data is small
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        print(f" Using {n_splits}-Fold CV")
    
    # Store results from all folds
    all_importances = []
    all_predictions = []
    all_actuals = []
    fold_scores = []
    
    fold = 0
    for train_idx, test_idx in cv.split(X):
        fold += 1
        print(f"\n  Processing fold {fold}/{n_splits}")
        
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = Y.iloc[train_idx], Y.iloc[test_idx]
        
        # Train model with careful parameters for small data
        model = CatBoostRegressor(
            iterations=200,
            learning_rate=0.05,
            depth=3,  # Shallow trees to prevent overfitting
            loss_function='RMSE',
            random_state=42,
            verbose=0  # Silent training
        )
        
        model.fit(X_train, y_train)
        
        # Get feature importance from this fold
        importance = model.get_feature_importance()
        all_importances.append(importance)
        
        # Make predictions and calculate score
        y_pred = model.predict(X_test)
        all_predictions.extend(y_pred)
        all_actuals.extend(y_test)
        
        fold_rmse = np.sqrt(np.mean((y_pred - y_test) ** 2))
        fold_scores.append(fold_rmse)
        
        print(f"   Fold {fold} RMSE: {fold_rmse:.4f}")
        print(f"   Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    return process_cv_results(all_importances, all_predictions, all_actuals, 
                            fold_scores, X.columns, cv_method)

def process_cv_results(all_importances, all_predictions, all_actuals, 
                      fold_scores, feature_names, cv_method):
    """Process and analyze cross-validation results"""
    
    print(f"\n  Processing {cv_method.upper()} Results...")
    
    # Convert to numpy array for easier calculations
    importance_array = np.array(all_importances)
    
    # Calculate overall performance
    overall_rmse = np.sqrt(np.mean((np.array(all_predictions) - np.array(all_actuals)) ** 2))
    mean_fold_rmse = np.mean(fold_scores)
    std_fold_rmse = np.std(fold_scores)
    
    print(f"  Overall CV Performance:")
    print(f"   Overall RMSE: {overall_rmse:.4f}")
    print(f"   Mean Fold RMSE: {mean_fold_rmse:.4f} ± {std_fold_rmse:.4f}")
    
    # Calculate feature importance statistics
    mean_importance = np.mean(importance_array, axis=0)
    std_importance = np.std(importance_array, axis=0)
    cv_importance = std_importance / (mean_importance + 1e-8)  # Coefficient of variation
    
    # Create comprehensive results dataframe
    results_df = pd.DataFrame({
        'Feature': feature_names,
        'Mean_Importance': mean_importance,
        'Std_Importance': std_importance,
        'CV_Importance': cv_importance,  # Lower is more stable
        'Percentage': (mean_importance / np.sum(mean_importance)) * 100,
        'Rank': np.argsort(np.argsort(-mean_importance)) + 1  # Rank (1 is most important)
    }).sort_values('Mean_Importance', ascending=False)
    
    return results_df, {
        'overall_rmse': overall_rmse,
        'mean_fold_rmse': mean_fold_rmse,
        'std_fold_rmse': std_fold_rmse,
        'importance_array': importance_array
    }

# =============================================================================
# 3. Visualization
# =============================================================================

def plot_cv_feature_importance(results_df, cv_info):
    """Create comprehensive visualization of CV feature importance"""
    
    print("\n  Creating visualizations...")
    
    # Create figure with subplots
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(20, 12))
    
    # Plot 1: Mean Feature Importance
    top_features = results_df.head(10)
    y_pos = np.arange(len(top_features))
    
    ax1.barh(y_pos, top_features['Mean_Importance'], xerr=top_features['Std_Importance'],
             capsize=5, alpha=0.7, color='skyblue')
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels(top_features['Feature'])
    ax1.set_xlabel('Feature Importance Score')
    ax1.set_title('Mean Feature Importance with Standard Deviation', fontweight='bold')
    ax1.invert_yaxis()
    ax1.grid(axis='x', alpha=0.3)
    
    # Plot 2: Importance Percentage
    ax2.pie(top_features['Percentage'], labels=top_features['Feature'], 
            autopct='%1.1f%%', startangle=90)
    ax2.set_title('Feature Importance Distribution (%)', fontweight='bold')
    
    # Plot 3: Stability (Coefficient of Variation)
    stable_features = results_df.nsmallest(10, 'CV_Importance')
    y_pos_stable = np.arange(len(stable_features))
    
    ax3.barh(y_pos_stable, stable_features['CV_Importance'], alpha=0.7, color='lightgreen')
    ax3.set_yticks(y_pos_stable)
    ax3.set_yticklabels(stable_features['Feature'])
    ax3.set_xlabel('Coefficient of Variation (Lower = More Stable)')
    ax3.set_title('Feature Importance Stability', fontweight='bold')
    ax3.invert_yaxis()
    ax3.grid(axis='x', alpha=0.3)
    
    # Plot 4: Performance Summary
    metrics = ['Overall RMSE', 'Mean Fold RMSE', 'Fold RMSE Std']
    values = [cv_info['overall_rmse'], cv_info['mean_fold_rmse'], cv_info['std_fold_rmse']]
    
    bars = ax4.bar(metrics, values, color=['lightcoral', 'lightblue', 'lightyellow'])
    ax4.set_ylabel('RMSE Score')
    ax4.set_title('Cross-Validation Performance Summary', fontweight='bold')
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{value:.4f}', ha='center', va='bottom')
    
    ax4.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    # Additional: Feature importance across folds (if enough folds)
    if cv_info['importance_array'].shape[0] > 3:
        plot_importance_across_folds(cv_info['importance_array'], results_df['Feature'])

def plot_importance_across_folds(importance_array, feature_names):
    """Plot how feature importance changes across folds"""
    
    plt.figure(figsize=(15, 8))
    
    # Plot top 5 features across folds
    top_5_indices = np.argsort(-np.mean(importance_array, axis=0))[:5]
    
    for idx in top_5_indices:
        plt.plot(range(1, len(importance_array) + 1), 
                importance_array[:, idx], 
                marker='o', linewidth=2, 
                label=feature_names[idx])
    
    plt.xlabel('Fold Number')
    plt.ylabel('Feature Importance Score')
    plt.title('Feature Importance Stability Across CV Folds', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, len(importance_array) + 1))
    plt.tight_layout()
    plt.show()






# Main Execution


if __name__ == "__main__":
    
    # 1. Load data
    X, Y = load_data()
    if X is None:
        exit()
    
    # 2. Choose CV method based on data size
    if len(X) <= 10:
        # Use LOOCV for very small datasets
        results_df, cv_info = cross_validation_feature_importance(X, Y, cv_method='loocv')
    else:
        # Use K-Fold for larger datasets
        results_df, cv_info = cross_validation_feature_importance(X, Y, cv_method='kfold', n_splits=5)
    
    # 3. Display results
    print("\n" + "="*50)
    print(" CROSS-VALIDATION FEATURE IMPORTANCE RESULTS")
    print("="*50)
    print(results_df.to_string(float_format='%.4f', index=False))
    
    # 4. Create visualizations
    plot_cv_feature_importance(results_df, cv_info)
    
  
    