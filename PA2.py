import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from scipy.stats import chi2
import plotly
import plotly.figure_factory as ff
import plotly.express as px
import chart_studio.plotly as py

data = pd.read_csv('trainFeatures.csv', header = None, nrows = None).to_numpy()
labels = data[:,0].astype(np.uint8)
features = data[:,1:]

standard_scalar = StandardScaler()
norm_features = standard_scalar.fit_transform(features)
df = pd.DataFrame(norm_features)
df['Labels'] = labels

# Outlier removal
def wilks_oulier_removal(x, p):
    m = x.mean(axis = 0)
    cov_i = np.linalg.inv(np.cov(x, rowvar=False))
    diff = x-m

    mahal = np.diag(diff @ cov_i @ diff.T)
    cutoff = chi2.ppf(p,x.shape[1])

    return np.argwhere(mahal>cutoff)
features_woutliers = norm_features.copy()
labels_woutliers = labels.copy()
unique_lables = np.unique(labels_woutliers)
percentile = 0.997
for l in unique_lables:
    class_inices = np.where(labels_woutliers == l)[0]

    class_outliers = wilks_oulier_removal(features_woutliers[class_inices], 0.997)
    features_woutliers = np.delete(features_woutliers,class_outliers, axis = 0)
    labels_woutliers = np.delete(labels_woutliers, class_outliers, axis = 0)

print(f'{norm_features.shape[0] - features_woutliers.shape[0]} outliers were found')
print(f'{features_woutliers.shape[0]} observations remain')

# Feature ranking using Bhattacharyya Distance
def feature_ranking(x, y):
    n_obs, n_feat = x.shape
    classes = np.unique(y)
    dist = np.zeros((n_feat,), dtype=np.float64)

    for f in range(n_feat):
        for c in classes:
            cl = x[np.where(y == c)[0], f]
            m_cl = np.mean(cl)
            s_cl = np.std(cl, ddof=1)

            other_cl = x[np.where(y != c)[0], f]
            m_other = np.mean(other_cl)
            s_other = np.std(other_cl, ddof=1)

            dist[f] += ((1/8)*2*(m_cl-m_other)*(2/(s_cl+s_other))) + ((1/2)*np.log((s_cl**2 + s_other**2)/(2*s_cl*s_other)))

    zero_one_scalar = MinMaxScaler()
    dist = zero_one_scalar.fit_transform(dist.reshape((-1,1)))[:,0]

    return dist

feature_ranks =feature_ranking(features_woutliers, labels_woutliers)
sorted_feature_ranks = np.flip(np.sort(feature_ranks))
sorted_feature_ranks_indices = np.flip(np.argsort(feature_ranks))

feat_df = pd.DataFrame()
feat_df['Features'] = [f'{i}' for i in sorted_feature_ranks_indices]
feat_df['Rankings'] = sorted_feature_ranks

top_20_features = sorted_feature_ranks_indices[:20]
top_features = features_woutliers[:,top_20_features]

print(f'Top 20 features have been kept:{top_20_features}')

# K fold cross validation
five_fold = KFold(n_splits= 5, shuffle = True, random_state=324)

def viz_accuracy(classification_res, model_name, normalize=True):
    classes = np.unique(classification_res[0][0])
    conf_mx = []
    for cls_true, cls_pred in classification_res:
        if cls_true.shape != cls_pred.shape:
            raise ValueError("must have same shape")
        if normalize:
            conf_mxt = confusion_matrix(y_true=cls_true, y_pred=cls_pred, normalize = "true")*100
        else:
            conf_mxt = confusion_matrix(y_true=cls_true, y_pred=cls_pred)
        conf_mx.append(conf_mxt)
    avg_conf_mtx = np.mean(np.array(conf_mx), axis = 0)
    text_avg_cof_mtx = [[str(round(y, 3)) for y in x] for x in avg_conf_mtx]

    if normalize:
        classification_acc = np.diag(avg_conf_mtx).mean()
    else:
        classification_acc = np.diag(avg_conf_mtx).sum() / len(classification_res[0][0])*100
    conf_fig = ff.create_annotated_heatmap(
        z = avg_conf_mtx,
        x = list(classes),
        y = list(classes),
        colorscale="BuGn",
        showscale = True,
        annotation_text=text_avg_cof_mtx)
    conf_fig.update_xaxes(title_text ="<b>Predicted</b>", constrain = "domain")
    conf_fig.update_yaxes(autorange="reversed", tickangle=-90, scaleanchor= "x", scaleratio = 1, title_text="<b>True</b>", constrain = "domain" )
    conf_fig.update_layout(title_text=f'{model_name}<br><b>Prediction Accuracy: {classification_acc:.2f}%</b>')
    conf_fig["layout"]["xaxis"].update(side="bottom")

    return conf_fig

class BayesClassifier:

    def __init__(self):
        self.train_prob_membership = None
        self.train_variances = None
        self.train_means = None
        self.num_features = None
        self.num_classes = None

    def train(self, x, y):
        classes = np.unique(y)

        self.num_classes = len(classes)
        self.num_features = x.shape[1]
        self.train_means = np.zeros((self.num_classes, self.num_features))
        self.train_variances = np.zeros((self.num_classes, self.num_features))
        self.train_prob_membership = np.zeros(self.num_classes)

        for idx, cls in enumerate(classes):
            self.train_means[idx] = x[np.where(y ==cls)].mean(axis =0)
            self.train_variances[idx] = x[np.where(y == cls)].var(axis=0)
            self.train_prob_membership[idx] = len(np.where(y==cls)[0])/len(y)

    def predict_prob(self, x):
        num_obs, num_feats = x.shape

        probs = np.zeros((num_obs, self.num_classes))
        for obs in range(num_obs):
            for cls in range(self.num_classes):
                p = self.train_prob_membership[cls]
                t1 = 1/np.sqrt(2*np.pi*self.train_variances[cls])
                t2 = (x[obs]-self.train_means[cls])**2 / self.train_variances[cls]
                t3 = np.exp(-0.5*t2)
                t4 = (t1*t3).prod()
                probs[obs,cls] = t4 *p
            probs[obs,:] = probs[obs,:]/np.sum(probs[obs,:])
        return probs

    def predict(self,x):
        return np.argmax(self.predict_prob(x), axis = 1)

classification_results = []
for train_ind, test_ind in five_fold.split(X=top_features, y=labels_woutliers):
    x_train, x_test = top_features[train_ind], top_features[test_ind]
    y_train, y_test = labels_woutliers[train_ind], labels_woutliers[test_ind]

    unique_classes = np.unique(y_test)

    gbc = BayesClassifier()
    gbc.train(x_train, y_train)
    bayes_pred = gbc.predict(x_test)

    classification_results.append((y_test, bayes_pred))

fig = viz_accuracy(classification_results, "Gaussian Bayes Classifier").show()
fig = viz_accuracy(classification_results, "Gaussian Bayes Classifier")
fig.write_html("conf.html")