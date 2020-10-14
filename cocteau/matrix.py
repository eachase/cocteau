__doc__ = "Tools for creating pandas DataFrames to store kilonovae data"
__author__ = "Eve Chase <eachase@lanl.gov>"

from astropy import units
import itertools
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier


# Matplotlib settings
import matplotlib.pyplot as plt


titles = {
    'morph' : 'Wind Morphology',
    'wind' : 'Wind Composition',
    'md' : r'Dynamical Ejecta Mass [$M_{\odot}$]',
    'vd' : r'Dynamical Ejecta Velocity [$c$]',
    'mw' : r'Wind Mass [$M_{\odot}$]',
    'vw' : r'Wind Velocity [$c$]',
    'angle': 'Angular Bin'
}

class KNMatrix(object):
    """
    Matrix of kilonova data   
    """
    def __init__(self, matrix):
        self.matrix = matrix
        self.corr_matrix = None

    def correlation(self):
        """
        Compute correlation matrix
        """
        self.corr_matrix = self.matrix.corr()
        return self.corr_matrix


class TimeBandMatrix(KNMatrix):
    """
    N-by-M matrix of magnitudes 
        N: kilonovae 
        M: timesteps for each band
    """
    def __init__(self, matrix, bandnames=None,
        times=None, knprops=None, time_units=units.day,
        mag_units=units.ABmag):

        self.matrix = matrix
        self.bandnames = bandnames
        self.times = times
        self.knprops = knprops
        self.time_units = time_units
        self.mag_units = mag_units


    def plot_corr(self):
        assert self.corr_matrix is not None

        corr = self.corr_matrix

        # Sort the correlation matrix
        for i, band in enumerate(self.bandnames):
            band_idx_arr = corr.columns[pd.Series(
                corr.columns).str.startswith(band[0])]
            if i == 0:
                combined_idx = band_idx_arr
            else:
                combined_idx = combined_idx.union(
                    band_idx_arr, sort=False)
        self.sorted_corr = corr.loc[combined_idx, combined_idx]

        # Set up plot
        ncol = len(self.sorted_corr.columns)
        nlabel = len(self.bandnames)
        label_offset = ncol/(2*nlabel)

        fig = plt.figure(figsize=(20,20))
        ax = fig.add_subplot(111)
        cax = ax.matshow(self.sorted_corr, cmap='coolwarm', 
            vmin=-1, vmax=1)
        cbar = fig.colorbar(cax)
        cbar.set_label('Correlation')
        ticks = np.linspace(label_offset, ncol - ncol/(
            2*nlabel), nlabel)

        for line in np.linspace(-0.5, ncol - 0.5, nlabel+1):
            ax.axvline(x=line, color='k', lw=5)
            ax.axhline(y=line, color='k', lw=5)

        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(self.bandnames)
        ax.set_yticklabels(self.bandnames)
        ax.set_ylim(ncol-0.5, -0.5)

        return ax

    def _set_axis_title(self, index):
        time = self.times[[int(index[1:])]].values[0]
        
        
        title = f'{index[0]}-band magnitude at {time:.3f} days (AB Mag)'
        return title


    def plot_corr_scatter(self, band1, time1, band2, time2, 
        prop=None):
        """
        Plot the scatter plot for a given point in 
        the correlation plot
        """
        
        # Define indices based on time and bands
        index1 = f'{band1[0]}{np.where(self.times == time1)[0][0]}'
        index2 = f'{band2[0]}{np.where(self.times == time2)[0][0]}'

        # Make a scatter plot of the two columns
        fig = plt.figure(figsize=(10,10))
        ax = fig.add_subplot(111)
        if prop is None:
            ax.scatter(self.matrix[index1], 
                self.matrix[index2])
            
        elif prop == 'v':
            cax = ax.scatter(self.matrix[index1], 
                self.matrix[index2], 
                c=self.knprops['vd']/self.knprops['vw'])
            cbar = fig.colorbar(cax)
            cbar.set_label('Ratio of Velocity of Dynamical Ejecta to Wind')
            
        elif prop == 'm':
            cax = ax.scatter(self.matrix[index1], 
                self.matrix[index2], 
                norm=matplotlib.colors.LogNorm(), 
                c=self.knprops['md']/self.knprops['mw'])
            cbar = fig.colorbar(cax)
            cbar.set_label('Ratio of Mass of Dynamical Ejecta to Wind')
            
        else:
            if prop == 'mw' or prop == 'md':
                kwargs = {'norm': matplotlib.colors.LogNorm()}
            else:
                kwargs = {}
            
            
            cax = ax.scatter(self.matrix[index1], 
                self.matrix[index2], 
                c=self.knprops[prop], **kwargs)
            cbar = fig.colorbar(cax)
            cbar.set_label(titles[prop])
            
        ax.set_xlim(-20, 10)
        ax.set_ylim(-20, 10)
            
        ax.set_xlabel(self._set_axis_title(index1))
        ax.set_ylabel(self._set_axis_title(index2))
     
        return ax



class TimeSpectralMatrix(KNMatrix):
    """
    N-by-M matrix of magnitudes 
        N: kilonovae 
        M: timesteps for each spectral bin
    """
    def __init__(self):
        self.wavelengths = None
        self.times = None
        self.angle = None



class MagMatrix(KNMatrix):
    """
    N-by-M matrix of magnitudes 
        N: kilonovae 
        M: bands
    """
    def __init__(self, matrix, bandnames=None,
        times=None, knprops=None, time_units=units.day,
        mag_units=units.ABmag):

        self.matrix = matrix
        self.bandnames = bandnames
        self.times = times
        self.knprops = knprops
        self.time_units = time_units
        self.mag_units = mag_units


    def split_by_time(self, time):
        """
        Split matrix into individual time chunks
        """
        assert time in self.times.values

        # Trim properties for a certain time
        magnitudes = self.matrix.loc[self.times == time]
        idx_for_time = np.asarray([i[0] for i in magnitudes.index])
        knprops = self.knprops.iloc[idx_for_time]

        return MagMatrixFixedTime(magnitudes, time,
            self.bandnames, knprops, self.time_units,
            self.mag_units)


    def to_timebandmatrix(self):
        """
        Restructure matrix as TimeBandMatrix object
        """


        lightcurves_per_time = []
        # Iterate over each event
        for idx_lc, lightcurves_per_band in self.matrix.groupby(
            level=0):

            mags_over_time = None
            bandnames = lightcurves_per_band.columns.values
            for j, row in lightcurves_per_band.iterrows():
                idx_time = j[1]

                # Set up new column names
                col_names = [f'{name[0]}{idx_time}' \
                    for name in bandnames]

                # Combine magnitudes into one row
                mags_new = pd.DataFrame(row.values.reshape(
                    1, len(col_names)), columns=col_names)
                if idx_time == 0:
                    mags_over_time = mags_new
                else:
                    mags_over_time = mags_over_time.join(mags_new)

            # Combine magnitudes for all light curves
            lightcurves_per_time.append(mags_over_time)

        # Store magnitudes in dataframe
        matrix = pd.concat(lightcurves_per_time, 
            sort=False).dropna(axis='columns')

        return TimeBandMatrix(matrix, bandnames=self.bandnames,
            times=self.times, knprops=self.knprops, 
            time_units=self.time_units, mag_units=self.mag_units)



    def pca_over_time(self):
        """
        Compute a PCA for each timestep
        """
        # Split into a TimeBandMatrix

        # Perform PCA
        pass   



class MagMatrixFixedTime(MagMatrix):
    """
    N-by-M matrix of magnitudes at a fixed time
        N: kilonovae
        M: bands
    """
    def __init__(self, matrix, time, bandnames=None,
        knprops=None, time_units=units.day, 
        mag_units=units.ABmag):

        self.matrix = matrix
        self.time = time * time_units
        self.bandnames = bandnames
        self.knprops = knprops
        self.mag_units = mag_units


    def fit_pca(self, n_components=2):
        """
        Perform principal component analysis
        """

        # FIXME: determine appropriate number of components

        self.pca = PCA(n_components=n_components) 
        self.projected_mags = self.pca.fit_transform(self.matrix)        

        # Check that the components are the correct sign
        if np.sum(self.pca.components_[0]) < 0:
            self.pca.components_[0] = -self.pca.components_[0]
            self.projected_mags[:,0] = -self.projected_mags[:,0]

        if n_components > 1:
            if self.pca.components_[1][0] > 0:
                self.pca.components_[1] = -self.pca.components_[1]
                self.projected_mags[:,1] = -self.projected_mags[:,1]




        return self.projected_mags


    def set_ncomponents(self):
        """
        Determine the number of principal components needed
        """
        pass



    def regression_accuracy(self, prop='vd', niter=5):
        """
        Record the typical credible interval at which 
        parameters are recovered using a random forest regressor
        """

        # Use magnitudes as inputs
        X = self.matrix.values

        # And properties as target labels
        y = np.log10(self.knprops[prop].values)

        # Split into testing and training sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2)
        
        # Set up array to store credible intervals
        num_samples = X_test.shape[0]
        credibleintervals = np.zeros((niter, num_samples))

        # For each iteration (to reduce ratty data)
        for i in np.arange(niter):
            # Split into testing and training sets
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2)
            
            # Set up a regressor
            rf = RandomForestRegressor(n_estimators=1000,
                max_depth=100, min_samples_split=20,
                min_samples_leaf=1, max_features='sqrt')

            # Fit the regressor
            # FIXME: change sklearn source code
            idx = rf.fit(X_train, y_train)

            # Predict the output for every value in testing set
            for j, mags in enumerate(X_test):
                # Evaluate the probability distribution
                pred_dist = np.asarray([dt.predict(mags.reshape(
                    1, -1))[0] for dt in rf.estimators_])

                # Record width of 90% credible interval
                credibleintervals[i][j] = np.percentile(
                    pred_dist, 95) - np.percentile(pred_dist, 5)

        return credibleintervals.flatten()


    def classification_accuracy(self, prop='vd', classifier='knn', 
        rs=None, report_err=False, niter=50):
        """
        Quantify how well we can classify individual properties
        """

        # Set up the dataset into inputs and targets
        X = self.matrix.values
        prop_values = self.knprops[prop].values
        unique_propvals = np.unique(prop_values)
        y = np.zeros_like(prop_values)
        for i, prop_val in enumerate(unique_propvals):
            y[np.where(prop_values == prop_val)] = i


        # Use a GridSearch to improve classification
        if not report_err:
            niter = 1
        fit_scores = np.zeros(niter)
        for i in np.arange(niter):

            # Split the data into train and test sets
            X_train, X_test, y_train, y_test = train_test_split(X, y,
                test_size=0.2, random_state=rs)
            

            if classifier == 'knn':
                param_grid = {'n_neighbors': np.arange(1,10)} 
                # FIXME: maybe some cases where a larger n_neighbors is preferred
                self.clf = GridSearchCV(KNeighborsClassifier(), param_grid, cv=5)
            elif classifier == 'tree':
                param_grid = {'criterion': ['gini','entropy'],
                              'max_depth': np.arange(4, 15)}
                self.clf = GridSearchCV(DecisionTreeClassifier(), param_grid, cv=5)
            else:
                raise ValueError("Enter knn or tree for classifier")            

            # Fit the classifier
            self.clf.fit(X_train, y_train)
                
            # Score classifier on test data
            fit_scores[i] = self.clf.score(X_test, y_test)

        # Return fit_scores statistics
        self.score_median = np.median(fit_scores)
        self.score_lower5 = np.percentile(fit_scores, 5)
        self.score_upper95 = np.percentile(fit_scores, 95)
         
        if report_err:   
            return self.score_median, self.score_lower5, self.score_upper95
        else:
            return self.score_median    


    def plot(self, prop='vd', textfont=25, dim=20):


        # Shift points over to reflect true mean
        ref_mean = np.mean(np.dot(self.matrix, 
            self.pca.components_.T), axis=0)
        shifted_components = np.zeros_like(self.projected_mags)
        for i in np.arange(len(ref_mean)):
            shifted_components[:,i] = self.projected_mags[:,i] + \
                ref_mean[i]



        # Make strings of PCA components
        comp = self.pca.components_
        pc1_str = ''
        pc2_str = ''
        for i, band in enumerate(self.bandnames):
            pc1_str += ('%.3f%s ' % (comp[0,i], band[0]))
            pc2_str += ('%.3f%s ' % (comp[1,i], band[0]))
            if i < len(self.bandnames) - 1:
                pc1_str += '+ '
                pc2_str += '+ '
       
        num_bands = comp.shape[1]


        if prop == 'mw' or prop == 'md':
            kwargs = {'norm': matplotlib.colors.LogNorm()}
        else:
            kwargs = {}


        fig, ax = plt.subplots(figsize=(12, 10))
        if prop == 'runnumber':
            # Set up filename labels
            filename_idx_arr = np.asarray(list(itertools.chain.from_iterable(
                itertools.repeat(x, 54) \
                for x in np.arange(int(self.knprops.shape[0] / 54)))))
            im = ax.scatter(shifted_components[:,0], shifted_components[:,1],
                c=filename_idx_arr, **kwargs)


        else:
            im = ax.scatter(shifted_components[:,0], shifted_components[:,1],
                c=self.knprops[prop], **kwargs)
        ax.set_xlabel('First Component')
        ax.set_ylabel('Second Component')
        ax.set_title(titles[prop])
        ax.set_xlim([-25-dim, -25+dim])
        ax.set_ylim([-dim, dim])
        plt.colorbar(im)
        
        plt.subplots_adjust(bottom=0.2, left=0.3)
        
        box_props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        ax.text(0.05, 0.90, ('Time %.3f days' % self.time.to(
            units.day).value), fontsize=25,
            transform=ax.transAxes, bbox=box_props)
        
        ax.text(-0.05, -0.20, (pc1_str), fontsize=textfont,
            transform=ax.transAxes)
        
        ax.text(-0.35, -0.05, (pc2_str), fontsize=textfont,
            transform=ax.transAxes, rotation=90)

        return ax


class CorrelationMatrix(object):
    """
    Square matrix of correlations
    """
    def __init__(self, matrix):
        self.matrix = matrix

    def plot(self):
        pass




