% Modified from 
% https://github.com/USNavalResearchLaboratory/TrackerComponentLibrary/blob/8eea2bd66aa6643dd0e704fc22428110d0923439/Mathematical%20Functions/Geometry/nearestPointInPolytope.m

function [x,full_w,norm_x,exitCode]=WolfeAlg(P,Z1,Z2,Z3,maxIter,indSub,subVec)
%%WOLFEALG This function implement's Wolfe's algorithm in [1] for finding
%          the point in a polytope that is closest to the origin.
%
%REFERENCES:
%[1] P. Wolfe, "Finding the nearest point in a polytope," Mathematical
%    Programming, vol. 11, no. 1, pp. 128-149, Dec. 1976.
%
%January 2018 David F. Crouse, Naval Research Laboratory, Washington D.C.

if nargin>5 && ~isempty(indSub):
    parfor i = 1:n
        P_new(:,i) = subVec;
        WolfeAlg(P_new,Z1,Z2,Z3,maxIter)
    end
end

n=size(P,1);
m=size(P,2);

%Indices of the columns in P that are in the columns constituting Q.
SIdx=zeros(m,1);
w=zeros(m,1);

PMags2=sum(P.*P,1);

bVec=zeros(n+1,1);
bVec(1)=1;

%Step 0
[~,J]=min(PMags2);
SIdx(1)=J;
numInSet=1;
w(1)=1;

exitCode=-1;
skip1=false;
for curIter=1:maxIter
    if(skip1==false)
        %Step 1a
        x=P(:,SIdx(1:numInSet))*w(1:numInSet);

        %Step 1b
        [~,J]=min(sum(bsxfun(@times,x,P),1));

        %Step 1c
        if(x'*P(:,J)>x'*x-Z1*max(PMags2(J),PMags2(SIdx(1:numInSet))))
            exitCode=0;
            break;
        end

        %Step 1d
        if(any(J==SIdx(1:numInSet)))
            exitCode=1;
            break;
        end

        %Step 1e
        numInSet=numInSet+1;
        SIdx(numInSet)=J;
        w(numInSet)=0;
    end
    
    %Step 2a. We are solving Equation 4.1 using Algorithm C.
    u=[ones(1,numInSet);
       P(:,SIdx(1:numInSet))]\bVec;
    v=u/sum(u);

    %Step 2b
    if(any(v>Z2))
        w=v;
        skip1=false;
        continue;
    end
    
    %Step 3a
    POS=find(w(1:numInSet)-v>Z3);
    
    %Step3b
    theta=min(1,min(w(POS)./(w(POS)-v(POS))));
    
    %Step 3c
    w(1:numInSet)=theta*w(1:numInSet)+(1-theta)*v;
    
    %Step 3d This only needs to work on w(1:numInSet), but is is simpler in
    %Matlab to do it over all of the w vector that was preallocated.
    w(w<=Z2)=0;
    
    %Step 3e We choose to delete the first zero component.
    idx=find(w(1:numInSet)==0,1);
    if(isempty(idx))
        exitCode=2;
        break;
    end
    w(idx)=w(numInSet);
    SIdx(idx)=SIdx(numInSet);
    numInSet=numInSet-1;
    skip1=true;
end

full_w = zeros(m,1);
full_w(SIdx(1:(numInSet))) = w;

norm_x = norm(x);

end

%LICENSE:
%
%The source code is in the public domain and not licensed or under
%copyright. The information and software may be used freely by the public.
%As required by 17 U.S.C. 403, third parties producing copyrighted works
%consisting predominantly of the material produced by U.S. government
%agencies must provide notice with such work(s) identifying the U.S.
%Government material incorporated and stating that such material is not
%subject to copyright protection.
%
%Derived works shall not identify themselves in a manner that implies an
%endorsement by or an affiliation with the Naval Research Laboratory.
%
%RECIPIENT BEARS ALL RISK RELATING TO QUALITY AND PERFORMANCE OF THE
%SOFTWARE AND ANY RELATED MATERIALS, AND AGREES TO INDEMNIFY THE NAVAL
%RESEARCH LABORATORY FOR ALL THIRD-PARTY CLAIMS RESULTING FROM THE ACTIONS
%OF RECIPIENT IN THE USE OF THE SOFTWARE.
