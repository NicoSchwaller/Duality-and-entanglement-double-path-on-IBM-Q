%% Read data

% Number of points (number of different couples alpha, theta)
nb_points=13;
% Number of measurements (of V_K,P_k,C for each point)
nb_measurements=10;

% Initialize data vectors
V=zeros(nb_points*nb_measurements,1);
D=zeros(nb_points*nb_measurements,1);
C=zeros(nb_points*nb_measurements,1);

V1=zeros(nb_measurements,1);
V2=zeros(nb_measurements,1);
V3=zeros(nb_measurements,1);
V4=zeros(nb_measurements,1);
V5=zeros(nb_measurements,1);
V6=zeros(nb_measurements,1);
V7=zeros(nb_measurements,1);
V8=zeros(nb_measurements,1);
V9=zeros(nb_measurements,1);
V10=zeros(nb_measurements,1);
V11=zeros(nb_measurements,1);
V12=zeros(nb_measurements,1);
V13=zeros(nb_measurements,1);
C1=zeros(nb_measurements,1);
C2=zeros(nb_measurements,1);
C3=zeros(nb_measurements,1);
C4=zeros(nb_measurements,1);
C5=zeros(nb_measurements,1);
C6=zeros(nb_measurements,1);
C7=zeros(nb_measurements,1);
C8=zeros(nb_measurements,1);
C9=zeros(nb_measurements,1);
C10=zeros(nb_measurements,1);
C11=zeros(nb_measurements,1);
C12=zeros(nb_measurements,1);
C13=zeros(nb_measurements,1);
D1=zeros(nb_measurements,1);
D2=zeros(nb_measurements,1);
D3=zeros(nb_measurements,1);
D4=zeros(nb_measurements,1);
D5=zeros(nb_measurements,1);
D6=zeros(nb_measurements,1);
D7=zeros(nb_measurements,1);
D8=zeros(nb_measurements,1);
D9=zeros(nb_measurements,1);
D10=zeros(nb_measurements,1);
D11=zeros(nb_measurements,1);
D12=zeros(nb_measurements,1);
D13=zeros(nb_measurements,1);

% Read data
for j=1:nb_points
j=14-j;

theta_i_str=num2str(14-j);

for f=1:nb_measurements
    f_str=num2str(f);
    load_file_interference = join(['Classical_VDC_point_',theta_i_str,'_interference_BS2_',f_str,'.txt']);
    load_file_state = join(['Classical_VDC_point_',theta_i_str,'_state_',f_str,'.txt']);

    vdc = load(load_file_interference);
    

    V((j-1)*nb_measurements+f)=vdc(:,2);

    vdc_state=load(load_file_state);
    C((j-1)*nb_measurements+f)=max(vdc_state(:,1));
    D((j-1)*nb_measurements+f)=max(vdc_state(:,3));
    
    if j==1
         V1(f)=vdc(:,2);
         C1(f)=vdc_state(:,1);
         D1(f)=vdc_state(:,3);
     end
      if j==2
         V2(f)=vdc(:,2);
         C2(f)=vdc_state(:,1);
         D2(f)=vdc_state(:,3);
      end
      if j==3
         V3(f)=vdc(:,2);
         C3(f)=vdc_state(:,1);
         D3(f)=vdc_state(:,3);
      end
      if j==4
         V4(f)=vdc(:,2);
         C4(f)=vdc_state(:,1);
         D4(f)=vdc_state(:,3);
      end
      if j==5
         V5(f)=vdc(:,2);
         C5(f)=vdc_state(:,1);
         D5(f)=vdc_state(:,3);
      end
      if j==6
         V6(f)=vdc(:,2);
         C6(f)=vdc_state(:,1);
         D6(f)=vdc_state(:,3);
      end
      if j==7
         V7(f)=vdc(:,2);
         C7(f)=vdc_state(:,1);
         D7(f)=vdc_state(:,3);
      end
      if j==8
         V8(f)=vdc(:,2);
         C8(f)=vdc_state(:,1);
         D8(f)=vdc_state(:,3);
      end
      if j==9
         V9(f)=vdc(:,2);
         C9(f)=vdc_state(:,1);
         D9(f)=vdc_state(:,3);
      end
      if j==10
         V10(f)=vdc(:,2);
         C10(f)=vdc_state(:,1);
         D10(f)=vdc_state(:,3);
      end
      if j==11
         V11(f)=vdc(:,2);
         C11(f)=vdc_state(:,1);
         D11(f)=vdc_state(:,3);
      end
      if j==12
         V12(f)=vdc(:,2);
         C12(f)=vdc_state(:,1);
         D12(f)=vdc_state(:,3);
      end
      if j==13
         V13(f)=vdc(:,2);
         C13(f)=vdc_state(:,1);
         D13(f)=vdc_state(:,3);
      end
end

end

% Mean value for each point
mean_V=[mean(V1),mean(V2),mean(V3),mean(V4),mean(V5),mean(V6),mean(V7),mean(V8),mean(V9),mean(V10),mean(V11),mean(V12),mean(V13)];
mean_C=[mean(C1),mean(C2),mean(C3),mean(C4),mean(C5),mean(C6),mean(C7),mean(C8),mean(C9),mean(C10),mean(C11),mean(C12),mean(C13)];
mean_D=[mean(D1),mean(D2),mean(D3),mean(D4),mean(D5),mean(D6),mean(D7),mean(D8),mean(D9),mean(D10),mean(D11),mean(D12),mean(D13)];

% Error bounds

sigmaV_1=sqrt(var(V1,1));
sigmaD_1=sqrt(var(D1,1));
sigmaC_1=sqrt(var(C1,1));
sigmaV_2=sqrt(var(V2,1));
sigmaD_2=sqrt(var(D2,1));
sigmaC_2=sqrt(var(C2,1));
sigmaV_3=sqrt(var(V3,1));
sigmaD_3=sqrt(var(D3,1));
sigmaC_3=sqrt(var(C3,1));
sigmaV_4=sqrt(var(V4,1));
sigmaD_4=sqrt(var(D4,1));
sigmaC_4=sqrt(var(C4,1));
sigmaV_5=sqrt(var(V5,1));
sigmaD_5=sqrt(var(D5,1));
sigmaC_5=sqrt(var(C5,1));
sigmaV_6=sqrt(var(V6,1));
sigmaD_6=sqrt(var(D6,1));
sigmaC_6=sqrt(var(C6,1));
sigmaV_7=sqrt(var(V7,1));
sigmaD_7=sqrt(var(D7,1));
sigmaC_7=sqrt(var(C7,1));
sigmaV_8=sqrt(var(V8,1));
sigmaD_8=sqrt(var(D8,1));
sigmaC_8=sqrt(var(C8,1));
sigmaV_9=sqrt(var(V9,1));
sigmaD_9=sqrt(var(D9,1));
sigmaC_9=sqrt(var(C9,1));
sigmaV_10=sqrt(var(V10,1));
sigmaD_10=sqrt(var(D10,1));
sigmaC_10=sqrt(var(C10,1));
sigmaV_11=sqrt(var(V11,1));
sigmaD_11=sqrt(var(D11,1));
sigmaC_11=sqrt(var(C11,1));
sigmaV_12=sqrt(var(V12,1));
sigmaD_12=sqrt(var(D12,1));
sigmaC_12=sqrt(var(C12,1));
sigmaV_13=sqrt(var(V13,1));
sigmaD_13=sqrt(var(D13,1));
sigmaC_13=sqrt(var(C13,1));

[x,y,z] = sphere(150);

figure
hold on
grid on
hold on
scatter3(V1,D1,C1,'o','MarkerEdgeColor','black','MarkerFaceColor','red');
hold on
scatter3(V2,D2,C2,'o','MarkerEdgeColor','black','MarkerFaceColor','red');
hold on
scatter3(V3,D3,C3,'o','MarkerEdgeColor','black','MarkerFaceColor','red');
hold on
scatter3(V4,D4,C4,'o','MarkerEdgeColor','black','MarkerFaceColor','red');
hold on
scatter3(V5,D5,C5,'o','MarkerEdgeColor','black','MarkerFaceColor','red');
hold on
scatter3(V6,D6,C6,'o','MarkerEdgeColor','black','MarkerFaceColor','red');
hold on
scatter3(V7,D7,C7,'o','MarkerEdgeColor','black','MarkerFaceColor','red');
hold on
scatter3(V8,D8,C8,'o','MarkerEdgeColor','black','MarkerFaceColor','red');
hold on
scatter3(V9,D9,C9,'o','MarkerEdgeColor','black','MarkerFaceColor','red');
hold on
scatter3(V10,D10,C10,'o','MarkerEdgeColor','black','MarkerFaceColor','red');
hold on
scatter3(V11,D11,C11,'o','MarkerEdgeColor','black','MarkerFaceColor','red');
hold on
scatter3(V12,D12,C12,'o','MarkerEdgeColor','black','MarkerFaceColor','red');
hold on
scatter3(V13,D13,C13,'o','MarkerEdgeColor','black','MarkerFaceColor','red');
hold on
hSurface=surf(3*sigmaV_1*(x)+mean_V(1),3*sigmaD_1*(y)+mean_D(1),3*sigmaC_1*(z)+mean_C(1));
set(hSurface,'FaceColor',[0.1 0.8 0.8], 'FaceAlpha',0.5,'EdgeAlpha', 0.2);
hSurface=surf(3*sigmaV_2*(x)+mean_V(2),3*sigmaD_2*(y)+mean_D(2),3*sigmaC_2*(z)+mean_C(2));
set(hSurface,'FaceColor',[0.1 0.8 0.8], 'FaceAlpha',0.5,'EdgeAlpha', 0.2);
hSurface=surf(3*sigmaV_3*(x)+mean_V(3),3*sigmaD_3*(y)+mean_D(3),3*sigmaC_3*(z)+mean_C(3));
set(hSurface,'FaceColor',[0.1 0.8 0.8], 'FaceAlpha',0.5,'EdgeAlpha', 0.2);
hSurface=surf(3*sigmaV_4*(x)+mean_V(4),3*sigmaD_4*(y)+mean_D(4),3*sigmaC_4*(z)+mean_C(4));
set(hSurface,'FaceColor',[0.1 0.8 0.8], 'FaceAlpha',0.5,'EdgeAlpha', 0.2);
hSurface=surf(3*sigmaV_5*(x)+mean_V(5),3*sigmaD_5*(y)+mean_D(5),3*sigmaC_5*(z)+mean_C(5));
set(hSurface,'FaceColor',[0.1 0.8 0.8], 'FaceAlpha',0.5,'EdgeAlpha', 0.2);
hSurface=surf(3*sigmaV_6*(x)+mean_V(6),3*sigmaD_6*(y)+mean_D(6),3*sigmaC_6*(z)+mean_C(6));
set(hSurface,'FaceColor',[0.1 0.8 0.8], 'FaceAlpha',0.5,'EdgeAlpha', 0.2);
hSurface=surf(3*sigmaV_7*(x)+mean_V(7),3*sigmaD_7*(y)+mean_D(7),3*sigmaC_7*(z)+mean_C(7));
set(hSurface,'FaceColor',[0.1 0.8 0.8], 'FaceAlpha',0.5,'EdgeAlpha', 0.2);
hSurface=surf(3*sigmaV_8*(x)+mean_V(8),3*sigmaD_8*(y)+mean_D(8),3*sigmaC_8*(z)+mean_C(8));
set(hSurface,'FaceColor',[0.1 0.8 0.8], 'FaceAlpha',0.5,'EdgeAlpha', 0.2);
hSurface=surf(3*sigmaV_9*(x)+mean_V(9),3*sigmaD_9*(y)+mean_D(9),3*sigmaC_9*(z)+mean_C(9));
set(hSurface,'FaceColor',[0.1 0.8 0.8], 'FaceAlpha',0.5,'EdgeAlpha', 0.2);
hSurface=surf(3*sigmaV_10*(x)+mean_V(10),3*sigmaD_10*(y)+mean_D(10),3*sigmaC_10*(z)+mean_C(10));
set(hSurface,'FaceColor',[0.1 0.8 0.8], 'FaceAlpha',0.5,'EdgeAlpha', 0.2);
hSurface=surf(3*sigmaV_11*(x)+mean_V(11),3*sigmaD_11*(y)+mean_D(11),3*sigmaC_11*(z)+mean_C(11));
set(hSurface,'FaceColor',[0.1 0.8 0.8], 'FaceAlpha',0.5,'EdgeAlpha', 0.2);
hSurface=surf(3*sigmaV_12*(x)+mean_V(12),3*sigmaD_12*(y)+mean_D(12),3*sigmaC_12*(z)+mean_C(12));
set(hSurface,'FaceColor',[0.1 0.8 0.8], 'FaceAlpha',0.5,'EdgeAlpha', 0.2);
hSurface=surf(3*sigmaV_13*(x)+mean_V(13),3*sigmaD_13*(y)+mean_D(13),3*sigmaC_13*(z)+mean_C(13));
set(hSurface,'FaceColor',[0.1 0.8 0.8], 'FaceAlpha',0.5,'EdgeAlpha', 0.2);
axis equal
axis([0 1 0 1 0 1])
view(135,20)
hold on

hSurface=surf(x,y,z);
set(hSurface,'FaceColor',[0.5 0.5 0.5], 'FaceAlpha',0.3,'EdgeAlpha', 0.2);
xlabel('V')
ylabel('D')
zlabel('C')
title('Double-path experiment')
