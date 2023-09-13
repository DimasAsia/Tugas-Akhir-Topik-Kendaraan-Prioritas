import pygame
from saiki import detect_objects


def createJOI():
        # Path to the input videos
        video_path = 'C:/Users/Acer/yolov4/video/video2.mp4'     

        pygame.init()
        width, height = 512, 512
        screen = pygame.display.set_mode((width, height))
        clock = pygame.time.Clock()

        signal = []
        jalan1 = 5
        jalan2 = '---'
        jalan3 = '---'
        jalan4 = '---'
        lampu_merah = 0
        lampu_kuning = 5


        
        
        
        
        
        
        '''density = 'low'
        condition = 'normal'
        vehicle_counter = 3
        green_light_duration = 5'''

        '''red_light_radius = 20
        red_light_position = (256, 256)  # Koordinat pusat lampu merah'''

        running = True
        perform_detection = True
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    

            font = pygame.font.Font(None, 20)
            # Gambar garis-garis dan elemen lainnya di sini
            screen.fill((255, 255, 255))
            
            pygame.draw.line(screen, (0, 0, 0), (0, 190), (170, 190), 2)
            pygame.draw.line(screen, (0, 0, 0), (0, 190), (0, 340), 2)
            pygame.draw.line(screen, (0, 0, 0), (0, 340), (170, 340), 2)
            pygame.draw.line(screen, (0, 0, 0), (170, 0), (170, 190), 2)
            pygame.draw.line(screen, (0, 0, 0), (170, 340), (170, 512), 2)
            pygame.draw.line(screen, (0, 0, 0), (320, 0), (320, 190), 2)
            pygame.draw.line(screen, (0, 0, 0), (320, 190), (512, 190), 2)
            pygame.draw.line(screen, (0, 0, 0), (320, 340), (512, 340), 2)
            pygame.draw.line(screen, (0, 0, 0), (320, 340), (320, 512), 2)
            
            pygame.draw.rect(screen, (0, 0, 0), (130, 150, 40, 40))
            text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
            screen.blit(text_jalan1, (130, 180))
            
            pygame.draw.rect(screen, (0, 0, 0), (130, 340, 40, 40)) 
            text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
            screen.blit(text_jalan2, (130, 370))
            
            pygame.draw.rect(screen, (0, 0, 0), (320, 150, 40, 40))
            text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
            screen.blit(text_jalan3, (320, 180))
            
            pygame.draw.rect(screen, (0, 0, 0), (320, 340, 40, 40))
            text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
            screen.blit(text_jalan4, (320, 370))

            pygame.display.flip()
            clock.tick(1)

            if type(jalan1) == int:
                lampu_merah = jalan1 + lampu_kuning
                jalan1 -= 1
                
                text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                screen.blit(text_jalan1, (130, 180))
                
                lampu_merah -= 1
                text_merah = font.render(str(lampu_merah), True, (255, 255, 255))
                screen.blit(text_merah, (130, 370))
                
                if perform_detection:
                    density, condition, vehicle_counter, green_light_duration = detect_objects(video_path)
                    perform_detection = False
                    
                if jalan1 < 0:
                    lampu_kuning -= 1
                    text_kuning = font.render(str(lampu_kuning), True, (255, 255, 255))
                    screen.blit(text_kuning, (130, 180))
                    if lampu_kuning < 0:
                        jalan2 = green_light_duration
                        jalan1 = '---'
                        text_jalan1 = font.render(str(jalan1), True, (255, 255, 255))
                        screen.blit(text_jalan1, (130, 180))
                        
                
                
            elif type(jalan2) == int:
                lampu_merah = jalan2 + lampu_kuning
                jalan2 -= 1
                
                text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                screen.blit(text_jalan2, (130, 370))
                
                lampu_merah -= 1
                text_merah = font.render(str(lampu_merah), True, (255, 255, 255))
                screen.blit(text_merah, (320, 180))
                jalan3 = 10
                if jalan2 < 0:
                    lampu_kuning -= 1
                    text_kuning = font.render(str(lampu_kuning), True, (255, 255, 255))
                    screen.blit(text_kuning, (130, 370))
                    if lampu_kuning < 0:
                        jalan2 = '---'
                        text_jalan2 = font.render(str(jalan2), True, (255, 255, 255))
                        screen.blit(text_jalan2, (130, 370))
                
                
                
            elif type(jalan3) == int:
                lampu_merah = jalan3 +lampu_kuning
                jalan3 -= 1
                
                text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                screen.blit(text_jalan3, (320, 180))
                
                lampu_merah -= 1
                text_merah = font.render(str(lampu_merah), True, (255, 255, 255))
                screen.blit(text_merah, (320, 370))
                jalan4 = 15
                if jalan3 < 0:
                    lampu_kuning -= 1
                    text_kuning = font.render(str(lampu_kuning), True, (255, 255, 255))
                    screen.blit(text_jalan3, (320, 180))
                    if lampu_kuning < 0:
                        jalan3 = '---'
                        text_jalan3 = font.render(str(jalan3), True, (255, 255, 255))
                        screen.blit(text_jalan3, (320, 180))
                    
                
            elif type(jalan4) == int:
                lampu_merah = jalan4 +lampu_kuning
                jalan4 -= 1
                text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
                screen.blit(text_jalan4, (320, 370))
                
                lampu_merah -= 1
                text_merah = font.render(str(lampu_merah), True, (255, 255, 255))
                screen.blit(text_merah, (130, 180))
                jalan1 = 10
                if jalan4 < 0:
                    lampu_kuning -= 1
                    text_kuning = font.render(str(lampu_kuning), True, (255, 255, 255))
                    screen.blit(text_kuning, (320, 370))
                    if lampu_kuning < 0:
                        jalan4 = '---'
                        text_jalan4 = font.render(str(jalan4), True, (255, 255, 255))
                        screen.blit(text_jalan4, (320, 370))
                
   
            

            '''if jalan1 == 0 or jalan2 == 0 or jalan3 == 0 or jalan4 == 0:
                break

            if jalan1 < 0:
                density, condition, vehicle_counter, green_light_duration = detect_objects(video_path)
                if density == 0:
                    density = "kosong"
                elif density == 1:
                    density = "sepi"
                elif density == 2:
                    density = "sedang"
                elif density == 3:
                    density = "ramai"
                jalan1 = green_light_duration
                
            if jalan2 < 0:
                density, condition, vehicle_counter, green_light_duration = detect_objects(video_path)
                if density == 0:
                    density = "kosong"
                elif density == 1:
                    density = "sepi"
                elif density == 2:
                    density = "sedang"
                elif density == 3:
                    density = "ramai"
                jalan2 = green_light_duration
                
            if jalan3 < 0:
                density, condition, vehicle_counter, green_light_duration = detect_objects(video_path)
                if density == 0:
                    density = "kosong"
                elif density == 1:
                    density = "sepi"
                elif density == 2:
                    density = "sedang"
                elif density == 3:
                    density = "ramai"
                jalan3 = green_light_duration
                
            if jalan4 < 0:
                density, condition, vehicle_counter, green_light_duration = detect_objects(video_path)
                if density == 0:
                    density = "kosong"
                elif density == 1:
                    density = "sepi"
                elif density == 2:
                    density = "sedang"
                elif density == 3:
                    density = "ramai"
                jalan4 = green_light_duration'''
                

            
                  
            
            
            '''text_density = font.render(f'Kepadatan: {density}', True, (0, 0, 0))
            screen.blit(text_density, (10, 20))
            text_condition = font.render(f'Kondisi: {condition}', True, (0, 0, 0))
            screen.blit(text_condition, (10, 60))
            text_vehicle_counter = font.render(f'Kendaraan: {vehicle_counter}', True, (0, 0, 0))
            screen.blit(text_vehicle_counter, (330, 60))
            text_green_light_duration = font.render(f'Durasi Lampu Hijau: {green_light_duration}', True, (0, 0, 0))
            screen.blit(text_green_light_duration, (330, 20))'''
            
            
            
            'pygame.draw.circle(screen, (255, 0, 0), red_light_position, red_light_radius)'

            
        pygame.quit()

createJOI()
