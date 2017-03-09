function [blad_svm, idx_m_svm, ocena_jakosci_svm, dokladnosc_svm, czulosc_svm, specyficznosc_svm, blad_wazony_svm, dokladnosc_svm_sr, czulosc_svm_sr, specyficznosc_svm_sr, blad_wazony_svm_sr] = k_fold_svm(dane_uczace, status_uczacy, k, liczba_cech, prog)

warning off

options = statset('MaxIter',100000,'Display','off', 'UseParallel', true);
c = cvpartition(status_uczacy,'KFold',k);

blad_svm = zeros(k,50);
idx_m_svm = zeros(size(dane_uczace,1),k);
ocena_jakosci_svm = zeros(2,2);

for i=1:k
    idx = training(c,i);
    dane_treningowe = dane_uczace(:,idx);
    status_treningowy = status_uczacy(idx);
    dane_walidacyjne = dane_uczace(:,~idx);
    status_walidacyjny = status_uczacy(~idx);
    
    
	
	idx1 = rankfeatures(dane_treningowe, status_treningowy);
    idx_m_svm(:,i) = idx1;
    for b=1:liczba_cech
        disp(['SVM - iteracja: ' num2str(i) ', cecha: ' num2str(b)]);
        
        j = 0;
        for boxconstraint=-2:0.2:0
            j = j + 1;
            model = svmtrain(dane_treningowe(idx1(1:b),:)', status_treningowy, 'boxconstraint', 10^boxconstraint, 'options', options);
            predykcja = svmclassify(model, dane_walidacyjne(idx1(1:b),:)');

            klasa = predykcja > prog;
            ocena_jakosci_svm = zeros(2,2);
            for a=1:length(status_walidacyjny)
                if (klasa(a,1) == 1 && status_walidacyjny(1,a) == 1)
                        ocena_jakosci_svm(1,1) = ocena_jakosci_svm(1,1) + 1;
                    elseif (klasa(a,1) == 0 && status_walidacyjny(1,a) == 0)
                        ocena_jakosci_svm(2,2) = ocena_jakosci_svm(2,2) + 1;
                    elseif (klasa(a,1) == 1 && status_walidacyjny(1,a) == 0)
                        ocena_jakosci_svm(1,2) = ocena_jakosci_svm(1,2) + 1;
                    else 
                        ocena_jakosci_svm(2,1) = ocena_jakosci_svm(2,1) + 1;
                end
            end
            
      dokladnosc_svm(i,b,j) = (ocena_jakosci_svm(1,1) + ocena_jakosci_svm(2,2)) / (ocena_jakosci_svm(1,1) + ocena_jakosci_svm(1,2) + ocena_jakosci_svm(2,1) + ocena_jakosci_svm(2,2));
      czulosc_svm(i,b,j) = ocena_jakosci_svm(1,1) / (ocena_jakosci_svm(1,1) + ocena_jakosci_svm(2,1));
      specyficznosc_svm(i,b,j) = ocena_jakosci_svm(2,2) / (ocena_jakosci_svm(2,2) + ocena_jakosci_svm(1,2));
      blad_wazony_svm(i,b,j) = 1 - ((czulosc_svm(i,b,j) + specyficznosc_svm(i,b,j)) / 2);
      
        end
    
    
        
    end  
    
    
end
   


dokladnosc_svm_sr = sum(dokladnosc_svm/k);
czulosc_svm_sr = sum(czulosc_svm/k);
specyficznosc_svm_sr = sum(specyficznosc_svm/k);
blad_wazony_svm_sr = sum(blad_wazony_svm/k);

       
end

    





