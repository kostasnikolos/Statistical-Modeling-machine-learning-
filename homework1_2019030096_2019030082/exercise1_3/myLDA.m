function A = myLDA(Samples, Labels, NewDim)
% Input:    
%   Samples: The Data Samples 
%   Labels: The labels that correspond to the Samples
%   NewDim: The New Dimension of the Feature Vector after applying LDA
    
	[NumSamples, NumFeatures] = size(Samples);
    NumLabels = length(Labels);
    if (NumSamples ~= NumLabels)
        fprintf('\nNumber of Samples are not the same with the Number of Labels.\n\n');
        exit
    end
    Classes = unique(Labels);
    NumClasses = length(Classes);  %The number of classes


    %Calculate the Global Mean
	m0 = sum(Samples, 1)/NumSamples;
    mu = zeros(NumClasses, NumFeatures);
    P = zeros(NumClasses, 1);
    Sw = zeros(NumFeatures);
    Sb = zeros(NumFeatures);
    for i = 1:NumClasses            % For each class i
	    %Find the necessary statistics
        
        class_samples = sum(Classes(i)==Labels);
        %Calculate the Class Prior Probability
	    P(i) = class_samples/NumSamples;
        %Calculate the Class Mean 
	    mu(i,:) = sum(Samples(Classes(i)==Labels,:), 1)/class_samples;
        %Calculate the Within Class Scatter Matrix
	    Sw = Sw + P(i)*cov(Samples(Classes(i)==Labels,:));
        %Calculate the Between Class Scatter Matrix
	    Sb = Sb + P(i)*(mu(i,:) - m0)'*(mu(i,:) - m0);
    end
    
    %Eigen matrix EigMat=inv(Sw)*Sb
    EigMat = inv(Sw)*Sb;
    
    %Perform Eigendecomposition
    [V, D] = eig(EigMat);
    
    %Select the NewDim eigenvectors corresponding to the top NewDim
    %eigenvalues (Assuming they are NewDim<=NumClasses-1)
    [~, ind] = sort(diag(D),1,'descend');
	%% You need to return the following variable correctly.
	% A = zeros(NumFeatures,NewDim);  % Return the LDA projection vectors
    A = V(:, ind(1:NewDim));
