function [blad_lr, idx_m_lr, ocena_jakosci_lr, dokladnosc_lr, czulosc_lr, specyficznosc_lr, blad_wazony_lr, dokladnosc_lr_sr, czulosc_lr_sr, specyficznosc_lr_sr, blad_wazony_lr_sr] = k_fold_lr(dane_uczace, status_uczacy, k, liczba_cech, ~)

warning off

c = cvpartition(status_uczacy,'KFold',k);

blad_lr = zeros(k,50);
idx_m_lr = zeros(size(dane_uczace,1),k);
ocena_jakosci_lr = zeros(2,2);

for i=1:k
    idx = training(c,i);
    dane_treningowe = dane_uczace(:,idx);
    status_treningowy = status_uczacy(idx);
    dane_walidacyjne = dane_uczace(:,~idx);
    status_walidacyjny = status_uczacy(~idx);
    
    
	
	idx1 = rankfeatures(dane_treningowe, status_treningowy);
    idx_m_lr(:,i) = idx1;
    for b=1:liczba_cech
        disp(['LR - iteracja: ' num2str(i) ', cecha: ' num2str(b)]);
        model = glmfit(dane_treningowe(idx1(1:b),:)', status_treningowy', 'binomial');
        predykcja = glmval(model, dane_walidacyjne(idx1(1:b),:)', 'logit');

    j = 0;
    for prog = 0.1:0.05:0.9
        j = j + 1;
        klasa = predykcja > prog;
        ocena_jakosci_lr = zeros(2,2);
        for a=1:length(status_walidacyjny)
            if (klasa(a,1) == 1 && status_walidacyjny(1,a) == 1)
                    ocena_jakosci_lr(1,1) = ocena_jakosci_lr(1,1) + 1;
                elseif (klasa(a,1) == 0 && status_walidacyjny(1,a) == 0)
                    ocena_jakosci_lr(2,2) = ocena_jakosci_lr(2,2) + 1;
                elseif (klasa(a,1) == 1 && status_walidacyjny(1,a) == 0)
                    ocena_jakosci_lr(1,2) = ocena_jakosci_lr(1,2) + 1;
                else 
                    ocena_jakosci_lr(2,1) = ocena_jakosci_lr(2,1) + 1;
            end
        end
      dokladnosc_lr(i,b,j) = (ocena_jakosci_lr(1,1) + ocena_jakosci_lr(2,2)) / (ocena_jakosci_lr(1,1) + ocena_jakosci_lr(1,2) + ocena_jakosci_lr(2,1) + ocena_jakosci_lr(2,2));
      czulosc_lr(i,b,j) = ocena_jakosci_lr(1,1) / (ocena_jakosci_lr(1,1) + ocena_jakosci_lr(2,1));
      specyficznosc_lr(i,b,j) = ocena_jakosci_lr(2,2) / (ocena_jakosci_lr(2,2) + ocena_jakosci_lr(1,2));
      blad_wazony_lr(i,b,j) = 1 - ((czulosc_lr(i,b,j) + specyficznosc_lr(i,b,j)) / 2);
    end
    
    
        
    end  
    

    
end

   
dokladnosc_lr_sr = sum(dokladnosc_lr/k);
czulosc_lr_sr = sum(czulosc_lr/k);
specyficznosc_lr_sr = sum(specyficznosc_lr/k);
blad_wazony_lr_sr = sum(blad_wazony_lr/k);

       
end

    





